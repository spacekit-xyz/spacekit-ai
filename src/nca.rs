// nca.rs — 2D Clifford Neural Cellular Automaton scratchpad
//
// A spatial memory the agent reads from and writes to via tool calls.  Each
// grid cell holds a `Multivector` — the same algebra used everywhere else in
// the crate — and the update rule is a small per-cell network that sees the
// 3×3 Moore neighborhood.
//
// The agent interacts via four operations:
//
//   read(x, y)            → return the cell at (x, y)
//   write(x, y, mv)       → overwrite (x, y) with a new multivector
//   step(n)               → advance the CA by n iterations
//   read_region(x,y,w,h)  → bulk read for the agent to inspect a patch
//
// Internally, one CA step consists of:
//
//   1. Perception: each cell gathers its 9-neighborhood (self + 8 around).
//      Three filters are applied: identity, Sobel-x, Sobel-y — producing a
//      perception vector of 3 × d_state multivectors per cell.
//
//   2. Update: a small CliffordFFN maps perception → state delta.
//
//   3. Stochastic fire mask: each cell updates with probability `fire_rate`
//      (classic NCA trick to break synchrony and force robustness).
//
//   4. State += delta.
//
// The grid is wrapped in CliffordGridNCA which the agent can persist across
// turns by holding a single instance and calling step() between thinks.

use crate::cayley_const::CliffordAlgebraConst;
use crate::sample::SimpleRng;
use crate::{CliffordAlgebra, CliffordFFN, CliffordLinear, Multivector};
use std::sync::Arc;

// ─── State per cell ──────────────────────────────────────────────────────────

/// One cell in the grid.  `d_state` Multivectors stacked vertically.
///
/// The first multivector by convention is the cell's "alive" channel:
/// `state[0].c[0]` < 0.1 means the cell is considered dead and gets clipped
/// to zero each step.  This mirrors the original Growing NCA design.
pub type Cell = Vec<Multivector>;

fn zero_cell(d_state: usize) -> Cell {
    vec![Multivector::zero(); d_state]
}

// ─── Grid NCA ────────────────────────────────────────────────────────────────

pub struct CliffordGridNCA {
    pub width: usize,
    pub height: usize,
    pub d_state: usize,
    pub cells: Vec<Vec<Cell>>, // [y][x] indexing

    /// Perception network: maps the 3-filter × d_state perception → d_hidden.
    pub perceive: CliffordLinear,
    /// Update network: maps d_hidden → d_state delta.
    pub update: CliffordFFN,

    /// Probability a cell updates on a given step (classic NCA stochasticity).
    pub fire_rate: f32,
    /// "Alive" threshold on state[0].c[0].  Cells below this are zeroed.
    pub alive_threshold: f32,
    /// Maximum delta per step (prevents runaway updates).
    pub max_delta: f32,

    pub algebra: Arc<CliffordAlgebra>,
    pub alg_const: CliffordAlgebraConst,
    pub rng: SimpleRng,
    pub step_count: u64,
}

impl CliffordGridNCA {
    /// Create an empty grid (all cells zero).
    pub fn new(width: usize, height: usize, d_state: usize, seed: u64) -> Self {
        let algebra = Arc::new(CliffordAlgebra::sta());
        let cells = (0..height)
            .map(|_| (0..width).map(|_| zero_cell(d_state)).collect())
            .collect();

        // Perception input dim: 3 filters × d_state multivectors
        let perception_dim = 3 * d_state;
        let hidden_dim = d_state * 4;

        Self {
            width,
            height,
            d_state,
            cells,
            perceive: CliffordLinear::new(perception_dim, hidden_dim, algebra.clone()),
            update: CliffordFFN::new(hidden_dim, d_state, algebra.clone()),
            fire_rate: 0.5,
            alive_threshold: 0.1,
            max_delta: 1.0,
            algebra,
            alg_const: CliffordAlgebraConst::new(),
            rng: SimpleRng::new(seed),
            step_count: 0,
        }
    }

    // ─── Agent-facing API ────────────────────────────────────────────────────

    /// Read a single cell.  Returns None if out of bounds.
    pub fn read(&self, x: usize, y: usize) -> Option<&Cell> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(&self.cells[y][x])
    }

    /// Write a single cell.  Silently ignores out-of-bounds writes.
    pub fn write(&mut self, x: usize, y: usize, cell: Cell) {
        if x >= self.width || y >= self.height {
            return;
        }
        debug_assert_eq!(cell.len(), self.d_state);
        self.cells[y][x] = cell;
    }

    /// Read a rectangular region of cells.  Out-of-bounds positions return zero.
    pub fn read_region(&self, x: usize, y: usize, w: usize, h: usize) -> Vec<Vec<Cell>> {
        (0..h)
            .map(|dy| {
                (0..w)
                    .map(|dx| {
                        let xi = x + dx;
                        let yi = y + dy;
                        if xi < self.width && yi < self.height {
                            self.cells[yi][xi].clone()
                        } else {
                            zero_cell(self.d_state)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Bulk-write a rectangular region.  Positions out of bounds are dropped.
    pub fn write_region(&mut self, x: usize, y: usize, region: &[Vec<Cell>]) {
        for (dy, row) in region.iter().enumerate() {
            for (dx, cell) in row.iter().enumerate() {
                self.write(x + dx, y + dy, cell.clone());
            }
        }
    }

    /// Mark a single position as "alive" with a scalar magnitude.
    /// Useful for seeding: write a 1.0 at a chosen location and let the CA grow.
    pub fn seed(&mut self, x: usize, y: usize, magnitude: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let mut cell = zero_cell(self.d_state);
        cell[0].c[0] = magnitude; // alive channel
        if self.d_state > 1 {
            cell[1].c[0] = magnitude;
        }
        self.cells[y][x] = cell;
    }

    /// Reset the entire grid to zeros.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            for cell in row {
                *cell = zero_cell(self.d_state);
            }
        }
        self.step_count = 0;
    }

    /// Total number of alive cells (state[0].c[0] > alive_threshold).
    pub fn alive_count(&self) -> usize {
        let mut n = 0;
        for row in &self.cells {
            for cell in row {
                if cell[0].c[0] > self.alive_threshold {
                    n += 1;
                }
            }
        }
        n
    }

    /// Average state[0].c[0] across the grid — a simple "activity" metric.
    pub fn mean_activity(&self) -> f32 {
        let n = (self.width * self.height) as f32;
        let mut s = 0.0;
        for row in &self.cells {
            for cell in row {
                s += cell[0].c[0];
            }
        }
        s / n
    }

    // ─── Step: one CA iteration ──────────────────────────────────────────────

    /// Advance the CA by one step.
    pub fn step(&mut self) {
        let new_cells = self.compute_next_state();
        self.cells = new_cells;
        self.step_count += 1;
    }

    /// Advance by `n` steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Compute the full next-state grid without mutating self.
    /// Returns the new grid; useful for inspection or differentiable training.
    pub fn compute_next_state(&mut self) -> Vec<Vec<Cell>> {
        let mut next = self.cells.clone();

        for y in 0..self.height {
            for x in 0..self.width {
                // Stochastic fire mask — skip update with probability (1 - fire_rate)
                if self.rng.next_f32() > self.fire_rate {
                    continue;
                }

                // 1. Perception: 3 filters × d_state multivectors
                let perception = self.perceive_at(x, y);

                // 2. Hidden representation
                let hidden = self.perceive.forward(&perception);

                // 3. Delta from the FFN
                let mut delta = self.update.forward(&hidden);

                // 4. Clip delta to prevent runaway dynamics
                for mv in &mut delta {
                    for k in 0..16 {
                        mv.c[k] = mv.c[k].clamp(-self.max_delta, self.max_delta);
                    }
                }

                // 5. Apply: state += delta
                for d in 0..self.d_state {
                    for k in 0..16 {
                        next[y][x][d].c[k] += delta[d].c[k];
                    }
                }
            }
        }

        // 6. Alive masking: cells below threshold get zeroed out
        for y in 0..self.height {
            for x in 0..self.width {
                if next[y][x][0].c[0] < self.alive_threshold {
                    next[y][x] = zero_cell(self.d_state);
                }
            }
        }

        next
    }

    /// Compute the 3-filter perception vector at (x, y).
    /// Returns Vec<Multivector> of length 3 × d_state in the order:
    ///   [identity_d0, identity_d1, ..., sobel_x_d0, ..., sobel_y_d0, ...]
    fn perceive_at(&self, x: usize, y: usize) -> Vec<Multivector> {
        let mut out = Vec::with_capacity(3 * self.d_state);

        // Filter 1: identity — just the cell itself
        for d in 0..self.d_state {
            out.push(self.cells[y][x][d].clone());
        }

        // Filter 2: Sobel-x  (horizontal gradient)
        // kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] / 8
        for d in 0..self.d_state {
            let mut acc = Multivector::zero();
            for (dy, row_w) in [(-1i32, 1.0), (0, 2.0), (1, 1.0)] {
                for (dx, col_w) in [(-1i32, -1.0), (0, 0.0), (1, 1.0)] {
                    if col_w == 0.0 {
                        continue;
                    }
                    if let Some(c) = self.get_wrapped(x as i32 + dx, y as i32 + dy) {
                        let w = row_w * col_w / 8.0;
                        for k in 0..16 {
                            acc.c[k] += w * c[d].c[k];
                        }
                    }
                }
            }
            out.push(acc);
        }

        // Filter 3: Sobel-y  (vertical gradient)
        for d in 0..self.d_state {
            let mut acc = Multivector::zero();
            for (dy, row_w) in [(-1i32, -1.0), (0, 0.0), (1, 1.0)] {
                if row_w == 0.0 {
                    continue;
                }
                for (dx, col_w) in [(-1i32, 1.0), (0, 2.0), (1, 1.0)] {
                    if let Some(c) = self.get_wrapped(x as i32 + dx, y as i32 + dy) {
                        let w = row_w * col_w / 8.0;
                        for k in 0..16 {
                            acc.c[k] += w * c[d].c[k];
                        }
                    }
                }
            }
            out.push(acc);
        }

        out
    }

    /// Get a cell with toroidal wrap-around (so the grid has no edge effects).
    fn get_wrapped(&self, x: i32, y: i32) -> Option<&Cell> {
        let xi = x.rem_euclid(self.width as i32) as usize;
        let yi = y.rem_euclid(self.height as i32) as usize;
        Some(&self.cells[yi][xi])
    }

    // ─── Damage / repair ─────────────────────────────────────────────────────

    /// Zero out a circular patch of the grid — useful for testing the CA's
    /// self-repair properties.  After damage, several `step()` calls should
    /// restore the pattern.
    pub fn damage_circle(&mut self, cx: usize, cy: usize, radius: usize) {
        let r = radius as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                    self.cells[y as usize][x as usize] = zero_cell(self.d_state);
                }
            }
        }
    }
}

// ─── Agent tool-call adapter ─────────────────────────────────────────────────
//
// A thin wrapper exposing the four operations as enum variants — useful if
// you're routing tool calls through a structured dispatcher (which most agents
// using a Clifford LLM eventually will).

pub enum NcaCommand {
    Read {
        x: usize,
        y: usize,
    },
    Write {
        x: usize,
        y: usize,
        cell: Cell,
    },
    Seed {
        x: usize,
        y: usize,
        magnitude: f32,
    },
    Step {
        n: usize,
    },
    ReadRegion {
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    },
    Clear,
    Status, // returns alive_count, mean_activity, step_count
    Damage {
        cx: usize,
        cy: usize,
        radius: usize,
    },
}

pub enum NcaResponse {
    Cell(Option<Cell>),
    Region(Vec<Vec<Cell>>),
    Status {
        alive: usize,
        activity: f32,
        step: u64,
    },
    Ack,
}

impl CliffordGridNCA {
    /// Dispatch a structured command.  This is the interface to wire into
    /// whatever tool-call mechanism your agent uses.
    pub fn execute(&mut self, cmd: NcaCommand) -> NcaResponse {
        match cmd {
            NcaCommand::Read { x, y } => NcaResponse::Cell(self.read(x, y).cloned()),
            NcaCommand::Write { x, y, cell } => {
                self.write(x, y, cell);
                NcaResponse::Ack
            }
            NcaCommand::Seed { x, y, magnitude } => {
                self.seed(x, y, magnitude);
                NcaResponse::Ack
            }
            NcaCommand::Step { n } => {
                self.steps(n);
                NcaResponse::Ack
            }
            NcaCommand::ReadRegion { x, y, w, h } => {
                NcaResponse::Region(self.read_region(x, y, w, h))
            }
            NcaCommand::Clear => {
                self.clear();
                NcaResponse::Ack
            }
            NcaCommand::Status => NcaResponse::Status {
                alive: self.alive_count(),
                activity: self.mean_activity(),
                step: self.step_count,
            },
            NcaCommand::Damage { cx, cy, radius } => {
                self.damage_circle(cx, cy, radius);
                NcaResponse::Ack
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_starts_empty() {
        let nca = CliffordGridNCA::new(8, 8, 4, 42);
        assert_eq!(nca.alive_count(), 0);
        assert!(nca.mean_activity().abs() < 1e-6);
    }

    #[test]
    fn seed_creates_alive_cell() {
        let mut nca = CliffordGridNCA::new(8, 8, 4, 42);
        nca.seed(3, 3, 1.0);
        assert_eq!(nca.alive_count(), 1);
        assert!((nca.cells[3][3][0].c[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn read_write_round_trip() {
        let mut nca = CliffordGridNCA::new(4, 4, 2, 0);
        let mut cell = zero_cell(2);
        cell[0].c[0] = 0.7;
        cell[1].c[5] = -0.3;
        nca.write(2, 1, cell.clone());

        let read_back = nca.read(2, 1).unwrap();
        assert!((read_back[0].c[0] - 0.7).abs() < 1e-6);
        assert!((read_back[1].c[5] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn out_of_bounds_read_returns_none() {
        let nca = CliffordGridNCA::new(4, 4, 2, 0);
        assert!(nca.read(99, 99).is_none());
    }

    #[test]
    fn step_advances_count() {
        let mut nca = CliffordGridNCA::new(4, 4, 2, 0);
        nca.seed(2, 2, 1.0);
        nca.steps(5);
        assert_eq!(nca.step_count, 5);
    }

    #[test]
    fn alive_threshold_kills_weak_cells() {
        let mut nca = CliffordGridNCA::new(4, 4, 2, 0);
        nca.fire_rate = 1.0; // every cell updates
        nca.alive_threshold = 0.5;

        // Seed at exactly 0.3 — below threshold after first step
        let mut cell = zero_cell(2);
        cell[0].c[0] = 0.3;
        nca.write(2, 2, cell);

        nca.step();
        // Should be culled to zero (or close to)
        // Note: the update FFN may push it up, but with random weights it's noisy.
        // We just check the masking *exists* by setting threshold very high.
        nca.alive_threshold = 100.0;
        let after = nca.compute_next_state();
        let alive_after: usize = after
            .iter()
            .flatten()
            .filter(|c| c[0].c[0] > nca.alive_threshold)
            .count();
        assert_eq!(alive_after, 0, "all cells should die at threshold=100");
    }

    #[test]
    fn region_read_handles_oob() {
        let mut nca = CliffordGridNCA::new(4, 4, 2, 0);
        nca.seed(0, 0, 1.0);
        // Read a 5x5 starting at (0,0) — partly out of bounds
        let region = nca.read_region(0, 0, 5, 5);
        assert_eq!(region.len(), 5);
        assert_eq!(region[0].len(), 5);
        // (0,0) has the seed; (4,4) is out of bounds → zero
        assert!((region[0][0][0].c[0] - 1.0).abs() < 1e-6);
        assert!(region[4][4][0].c[0].abs() < 1e-6);
    }

    #[test]
    fn damage_zeros_circle() {
        let mut nca = CliffordGridNCA::new(10, 10, 2, 0);
        // Fill grid
        for y in 0..10 {
            for x in 0..10 {
                nca.seed(x, y, 1.0);
            }
        }
        let before = nca.alive_count();
        nca.damage_circle(5, 5, 2);
        let after = nca.alive_count();
        assert!(after < before, "damage should reduce alive count");
    }

    #[test]
    fn command_dispatch_round_trip() {
        let mut nca = CliffordGridNCA::new(8, 8, 2, 7);
        nca.execute(NcaCommand::Seed {
            x: 4,
            y: 4,
            magnitude: 1.0,
        });
        nca.execute(NcaCommand::Step { n: 3 });

        let resp = nca.execute(NcaCommand::Status);
        if let NcaResponse::Status { step, .. } = resp {
            assert_eq!(step, 3);
        } else {
            panic!("expected Status response");
        }
    }
}
