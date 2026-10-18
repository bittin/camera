// SPDX-License-Identifier: GPL-3.0-only

//! Window-button visibility requested by the desktop, not inferred from screen size.

use ashpd::desktop::settings::Settings;
use cosmic::iced::futures::{SinkExt, StreamExt, channel::mpsc::Sender};

use super::Message;

const NAMESPACE: &str = "org.gnome.desktop.wm.preferences";
const KEY: &str = "button-layout";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowControls {
    pub minimize: bool,
    pub maximize: bool,
    pub close: bool,
}

impl Default for WindowControls {
    fn default() -> Self {
        Self {
            minimize: true,
            maximize: true,
            close: true,
        }
    }
}

impl WindowControls {
    pub(super) fn from_button_layout(layout: &str) -> Self {
        let buttons: Vec<_> = layout.split([':', ',']).map(str::trim).collect();
        Self {
            minimize: buttons.contains(&"minimize"),
            maximize: buttons.contains(&"maximize"),
            close: buttons.contains(&"close"),
        }
    }
}

/// Read through the portal so this also works inside Flatpak. The desktop's
/// preference is authoritative; do not infer it from hardware or window size.
pub(super) fn subscription() -> cosmic::iced::Subscription<Message> {
    super::subscription_with_id(
        "window-controls",
        cosmic::iced::stream::channel(8, async move |output| {
            match Settings::new().await {
                Ok(settings) => watch(settings, output).await,
                Err(err) => tracing::debug!(%err, "Window-button settings portal unavailable"),
            }
            // Keep the subscription alive on failure, retaining existing buttons.
            std::future::pending::<()>().await;
        }),
    )
}

async fn watch(settings: Settings, mut output: Sender<Message>) {
    // Subscribe first so a preference changed during the initial read is not lost.
    let changes = settings
        .receive_setting_changed_with_args::<String>(NAMESPACE, KEY)
        .await;
    if let Ok(layout) = settings.read::<String>(NAMESPACE, KEY).await
        && output
            .send(Message::WindowControlsChanged(
                WindowControls::from_button_layout(&layout),
            ))
            .await
            .is_err()
    {
        return;
    }
    if let Ok(mut changes) = changes {
        while let Some(layout) = changes.next().await {
            // Ignore malformed values rather than accidentally hiding controls.
            if let Ok(layout) = layout
                && output
                    .send(Message::WindowControlsChanged(
                        WindowControls::from_button_layout(&layout),
                    ))
                    .await
                    .is_err()
            {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires a running desktop settings portal with button-layout"]
    async fn live_portal_publishes_initial_window_controls() {
        let settings = Settings::new().await.unwrap();
        let expected = WindowControls::from_button_layout(
            &settings.read::<String>(NAMESPACE, KEY).await.unwrap(),
        );
        let mut stream = cosmic::iced::stream::channel(8, async move |output| {
            watch(settings, output).await;
        })
        .boxed();
        let message = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
            .await
            .expect("initial window-button preference")
            .expect("window-button message");
        let Message::WindowControlsChanged(actual) = message else {
            panic!("unexpected window-button message");
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn mobile_appmenu_layout_hides_all_window_buttons() {
        assert_eq!(
            WindowControls::from_button_layout("appmenu:"),
            WindowControls {
                minimize: false,
                maximize: false,
                close: false,
            }
        );
    }

    #[test]
    fn desktop_layouts_enable_only_explicit_button_names() {
        assert_eq!(
            WindowControls::from_button_layout(":minimize,maximize,close"),
            WindowControls::default()
        );
        assert_eq!(
            WindowControls::from_button_layout("close:appmenu,maximize"),
            WindowControls {
                minimize: false,
                maximize: true,
                close: true,
            }
        );
        for layout in ["", ":", "notclose:notmaximize,notminimize"] {
            let controls = WindowControls::from_button_layout(layout);
            assert!(!controls.minimize && !controls.maximize && !controls.close);
        }
    }
}
