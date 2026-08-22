//! Training CLI progress (stderr): overall phase bar + optional detail bar/spinner.
//! Log lines stay on stdout; bars use stderr so they do not fight `println!`.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::borrow::Cow;
use std::io::{stderr, IsTerminal};
use std::time::Duration;

/// Major train phases: positions `0 .. MAJOR_PHASE_COUNT-1`, then `finish_ok` sets full width.
pub const TRAIN_MAJOR_STEPS: u64 = 17;
pub const MAJOR_PHASE_COUNT: u64 = 16;

pub struct TrainUi {
    #[allow(dead_code)]
    multi: MultiProgress,
    overall: ProgressBar,
    detail: ProgressBar,
}

impl TrainUi {
    /// Returns `None` when disabled or stderr is not a TTY (CI, pipes).
    pub fn try_new(no_progress: bool) -> Option<Self> {
        if no_progress || !stderr().is_terminal() {
            return None;
        }
        let multi = MultiProgress::new();

        let overall = multi.add(ProgressBar::new(TRAIN_MAJOR_STEPS));
        overall.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{wide_bar:.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        overall.enable_steady_tick(Duration::from_millis(120));

        let detail = multi.add(ProgressBar::new_spinner());
        detail.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.yellow.dim} {wide_msg}")
                .unwrap(),
        );
        detail.set_draw_target(ProgressDrawTarget::hidden());

        Some(Self {
            multi,
            overall,
            detail,
        })
    }

    pub fn set_major_phase(&self, index: u64, label: impl Into<Cow<'static, str>>) {
        let i = index.min(MAJOR_PHASE_COUNT.saturating_sub(1));
        self.overall.set_position(i);
        self.overall.set_message(label);
    }

    pub fn detail_spinner(&self, msg: impl Into<Cow<'static, str>>) {
        self.detail.reset();
        self.detail.set_draw_target(ProgressDrawTarget::stderr());
        self.detail.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.yellow.dim} {wide_msg}")
                .unwrap(),
        );
        self.detail.enable_steady_tick(Duration::from_millis(80));
        self.detail.set_message(msg);
    }

    pub fn detail_bar(&self, total: u64, msg: impl Into<Cow<'static, str>>) {
        self.detail.reset();
        self.detail.set_draw_target(ProgressDrawTarget::stderr());
        self.detail.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.yellow} [{bar:28.yellow/black}] {human_pos}/{human_len} {wide_msg}",
                )
                .unwrap()
                .progress_chars("=> "),
        );
        self.detail.disable_steady_tick();
        self.detail.set_length(total.max(1));
        self.detail.set_position(0);
        self.detail.set_message(msg);
    }

    pub fn detail_inc(&self, n: u64) {
        self.detail.inc(n);
    }

    pub fn detail_finish_clear(&self) {
        self.detail.set_draw_target(ProgressDrawTarget::hidden());
        self.detail.reset();
        self.detail.set_message("");
    }

    pub fn finish_ok(&self, msg: impl Into<Cow<'static, str>>) {
        self.detail_finish_clear();
        self.overall.set_position(TRAIN_MAJOR_STEPS);
        self.overall.disable_steady_tick();
        self.overall.finish_with_message(msg.into().to_string());
    }
}
