// SPDX-License-Identifier: GPL-3.0-only

//! Provides localization support for this crate.

use i18n_embed::{
    DefaultLocalizer, LanguageLoader, Localizer,
    fluent::{FluentLanguageLoader, fluent_language_loader},
    unic_langid::LanguageIdentifier,
};
use rust_embed::RustEmbed;
use std::sync::LazyLock;

/// Applies the requested language(s) to requested translations from the `fl!()` macro.
pub fn init(requested_languages: &[LanguageIdentifier]) {
    init_localizer(&*localizer(), requested_languages);
}

fn init_localizer(localizer: &dyn Localizer, requested_languages: &[LanguageIdentifier]) {
    if let Err(why) = localizer.select(requested_languages) {
        tracing::error!(error = %why, "error while loading fluent localizations");
    }
}

// Get the `Localizer` to be used for localizing this library.
#[must_use]
pub fn localizer() -> Box<dyn Localizer> {
    Box::from(DefaultLocalizer::new(&*LANGUAGE_LOADER, &Localizations))
}

#[derive(RustEmbed)]
#[folder = "i18n/"]
struct Localizations;

pub static LANGUAGE_LOADER: LazyLock<FluentLanguageLoader> = LazyLock::new(|| {
    let loader: FluentLanguageLoader = fluent_language_loader!();

    loader
        .load_fallback_language(&Localizations)
        .expect("Error while loading fallback language");

    loader
});

/// Request a localized string by ID from the i18n/ directory.
#[macro_export]
macro_rules! fl {
    ($message_id:literal) => {{
        i18n_embed_fl::fl!($crate::i18n::LANGUAGE_LOADER, $message_id)
    }};

    ($message_id:literal, $($args:expr),*) => {{
        i18n_embed_fl::fl!($crate::i18n::LANGUAGE_LOADER, $message_id, $($args), *)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    struct FailingLocalizer;

    impl Localizer for FailingLocalizer {
        fn language_loader(&self) -> &dyn LanguageLoader {
            &*LANGUAGE_LOADER
        }

        fn i18n_assets(&self) -> &dyn i18n_embed::I18nAssets {
            &Localizations
        }

        fn select(
            &self,
            _requested_languages: &[LanguageIdentifier],
        ) -> Result<Vec<LanguageIdentifier>, i18n_embed::I18nEmbedError> {
            Err(i18n_embed::I18nEmbedError::RequestedLanguagesEmpty)
        }
    }

    struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for CaptureWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn localization_failures_use_the_application_subscriber() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let capture = bytes.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::ERROR)
            .with_writer(move || CaptureWriter(capture.clone()))
            .finish();
        tracing::subscriber::with_default(subscriber, || init_localizer(&FailingLocalizer, &[]));
        let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert!(
            output.contains("ERROR"),
            "missing error-level event: {output}"
        );
        assert!(output.contains("error while loading fluent localizations"));
        assert!(output.contains("The slice of requested languages cannot be empty."));
    }
}
