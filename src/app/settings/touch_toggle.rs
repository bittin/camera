// SPDX-License-Identifier: GPL-3.0-only

//! Touch-only adaptation for the pinned togglers' mouse-cursor hit testing.
//! Keep native painting/mouse behavior; capture presses while letting drags
//! reach the drawer scrollable, and activate only a matching release in view.

use cosmic::iced::advanced::{
    Clipboard, Layout, Shell, Widget, layout, renderer,
    widget::{Operation, Tree, tree},
};
use cosmic::iced::{Element, Event, Length, Point, Rectangle, Size, mouse, touch};

pub(super) struct TouchToggle<'a, Message, Theme, Renderer> {
    content: Element<'a, Message, Theme, Renderer>,
    on_tap: Option<Message>,
}

#[derive(Default)]
struct State {
    pressed: Option<(touch::Finger, Point)>,
}

impl<'a, Message, Theme, Renderer> TouchToggle<'a, Message, Theme, Renderer> {
    pub(super) fn new(
        content: impl Into<Element<'a, Message, Theme, Renderer>>,
        on_tap: Option<Message>,
    ) -> Self {
        Self {
            content: content.into(),
            on_tap,
        }
    }
}

impl<Message: Clone, Theme, Renderer: renderer::Renderer> Widget<Message, Theme, Renderer>
    for TouchToggle<'_, Message, Theme, Renderer>
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }
    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }
    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }
    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_mut(&mut self.content));
        if self.on_tap.is_none() {
            tree.state.downcast_mut::<State>().pressed = None;
        }
    }
    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }
    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits)
    }
    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }
    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        if let Event::Touch(event) = event {
            let state = tree.state.downcast_mut::<State>();
            let inside = |position| {
                // Scrollable translates the cursor and viewport into content
                // coordinates, but leaves touch event positions window-local.
                let position = position + layout.virtual_offset();
                layout.bounds().contains(position) && viewport.contains(position)
            };
            // Five logical pixels tolerate tap jitter, not a scrolling gesture.
            // Cancellation is permanent even if the finger returns to its origin.
            const TAP_SLOP: f32 = 5.0;
            match *event {
                touch::Event::FingerPressed { id, position } => {
                    if state.pressed.is_none() && self.on_tap.is_some() && inside(position) {
                        state.pressed = Some((id, position));
                        // The scrollable records touch_press_start even for a
                        // captured press. Leave moves uncaptured so it can
                        // start scrolling after its drag deadband, rather than
                        // converting a stationary tap's release to FingerLost.
                        shell.capture_event();
                    }
                }
                touch::Event::FingerMoved { id, position } => {
                    if let Some((finger, start)) = state.pressed
                        && finger == id
                        && (start.distance(position) > TAP_SLOP || !inside(position))
                    {
                        state.pressed = None;
                    }
                }
                touch::Event::FingerLifted { id, position } => {
                    if let Some((finger, start)) = state.pressed
                        && finger == id
                    {
                        state.pressed = None;
                        if inside(position)
                            && start.distance(position) <= TAP_SLOP
                            && let Some(message) = self.on_tap.as_ref()
                        {
                            shell.publish(message.clone());
                            shell.capture_event();
                        }
                    }
                }
                touch::Event::FingerLost { id, .. } => {
                    if state.pressed.is_some_and(|(finger, _)| finger == id) {
                        state.pressed = None;
                    }
                }
            }
            // Never pass touch through: both pinned togglers mistake the mouse
            // cursor for this finger (including a stale available mouse cursor).
            return;
        }
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }
    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            cursor,
            viewport,
        );
    }
    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        )
    }
}

impl<'a, Message: Clone + 'a, Theme: 'a, Renderer: renderer::Renderer + 'a>
    From<TouchToggle<'a, Message, Theme, Renderer>> for Element<'a, Message, Theme, Renderer>
{
    fn from(toggle: TouchToggle<'a, Message, Theme, Renderer>) -> Self {
        Self::new(toggle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic::iced::advanced::clipboard;

    #[derive(Clone, Debug, PartialEq)]
    enum Action {
        Toggle(bool),
        Scrolled,
    }

    fn replay(
        events: &[touch::Event],
        enabled: bool,
        cursor_available: bool,
        in_scrollable: bool,
    ) -> (Vec<Action>, Vec<bool>) {
        let native = cosmic::iced::widget::toggler::<Action, cosmic::Theme, ()>(false)
            .on_toggle_maybe(enabled.then_some(Action::Toggle as fn(bool) -> Action));
        let toggle = TouchToggle::new(native, enabled.then_some(Action::Toggle(true)));
        let mut widget: Element<'_, Action, cosmic::Theme, ()> = if in_scrollable {
            cosmic::iced::widget::scrollable(cosmic::iced::widget::column![
                toggle,
                cosmic::iced::widget::Space::new().height(300)
            ])
            .height(100)
            .on_scroll(|_| Action::Scrolled)
            .into()
        } else {
            toggle.into()
        };
        let mut tree = Tree::new(&widget);
        let node = widget.as_widget_mut().layout(
            &mut tree,
            &(),
            &layout::Limits::new(Size::ZERO, Size::new(100.0, 100.0)),
        );
        let viewport = Rectangle::with_size(Size::new(100.0, 100.0));
        let mut messages = Vec::new();
        let mut captured = Vec::new();
        for event in events {
            let position = match event {
                touch::Event::FingerPressed { position, .. }
                | touch::Event::FingerMoved { position, .. }
                | touch::Event::FingerLifted { position, .. }
                | touch::Event::FingerLost { position, .. } => *position,
            };
            let cursor = if cursor_available {
                mouse::Cursor::Available(position)
            } else {
                mouse::Cursor::Unavailable
            };
            let mut shell = Shell::new(&mut messages);
            widget.as_widget_mut().update(
                &mut tree,
                &Event::Touch(*event),
                Layout::new(&node),
                cursor,
                &(),
                &mut clipboard::Null,
                &mut shell,
                &viewport,
            );
            captured.push(shell.is_event_captured());
        }
        (messages, captured)
    }
    fn press(x: f32, y: f32) -> touch::Event {
        touch::Event::FingerPressed {
            id: touch::Finger(1),
            position: Point::new(x, y),
        }
    }
    fn lift(x: f32, y: f32) -> touch::Event {
        touch::Event::FingerLifted {
            id: touch::Finger(1),
            position: Point::new(x, y),
        }
    }
    fn moved(x: f32, y: f32) -> touch::Event {
        touch::Event::FingerMoved {
            id: touch::Finger(1),
            position: Point::new(x, y),
        }
    }

    #[test]
    fn motion_photo_touch_tap_in_drawer_scrollable_activates_on_release() {
        for available in [false, true] {
            let (messages, _) = replay(
                &[press(10.0, 10.0), lift(10.0, 10.0)],
                true,
                available,
                true,
            );
            assert_eq!(
                messages,
                [Action::Toggle(true)],
                "cursor available: {available}"
            );
        }
    }
    #[test]
    fn motion_photo_touch_drag_reaches_scrollable_without_toggling() {
        let (messages, _) = replay(
            &[press(10.0, 20.0), moved(10.0, 5.0), lift(10.0, 5.0)],
            true,
            true,
            true,
        );
        assert!(
            messages.contains(&Action::Scrolled),
            "must actually scroll: {messages:?}"
        );
        assert!(!messages.contains(&Action::Toggle(true)));
    }
    #[test]
    fn motion_photo_touch_cancelled_or_unmatched_gestures_never_activate() {
        for events in [
            vec![lift(10.0, 10.0)],
            vec![press(90.0, 90.0), lift(10.0, 10.0)],
            vec![press(10.0, 10.0), moved(10.0, 20.0), lift(10.0, 10.0)],
            vec![press(10.0, 10.0), lift(90.0, 90.0)],
            vec![
                press(10.0, 10.0),
                touch::Event::FingerLost {
                    id: touch::Finger(1),
                    position: Point::new(10.0, 10.0),
                },
                lift(10.0, 10.0),
            ],
            vec![
                press(10.0, 10.0),
                touch::Event::FingerLifted {
                    id: touch::Finger(2),
                    position: Point::new(10.0, 10.0),
                },
            ],
        ] {
            for available in [false, true] {
                assert!(
                    replay(&events, true, available, false).0.is_empty(),
                    "{events:?}"
                );
            }
        }
    }
    #[test]
    fn motion_photo_touch_tap_after_scrolling_uses_content_coordinates() {
        use cosmic::iced::Vector;
        let native = cosmic::iced::widget::toggler::<Action, cosmic::Theme, ()>(false)
            .on_toggle(Action::Toggle);
        let mut toggle = TouchToggle::new(native, Some(Action::Toggle(true)));
        let mut tree = Tree::new(&toggle as &dyn Widget<Action, cosmic::Theme, ()>);
        let node = toggle
            .layout(
                &mut tree,
                &(),
                &layout::Limits::new(Size::ZERO, Size::new(100.0, 100.0)),
            )
            .move_to(Point::new(0.0, 60.0));
        let layout = Layout::new(&node).with_virtual_offset(Vector::new(0.0, 50.0));
        // The scrollable passes a translated viewport, but finger coordinates
        // remain window-local. The switch is displayed at y=10, not y=60.
        let viewport = Rectangle {
            x: 0.0,
            y: 50.0,
            width: 100.0,
            height: 100.0,
        };
        let mut messages = Vec::new();
        for event in [press(10.0, 20.0), lift(10.0, 20.0)] {
            toggle.update(
                &mut tree,
                &Event::Touch(event),
                layout,
                mouse::Cursor::Unavailable,
                &(),
                &mut clipboard::Null,
                &mut Shell::new(&mut messages),
                &viewport,
            );
        }
        assert_eq!(messages, [Action::Toggle(true)]);
    }

    #[test]
    fn motion_photo_touch_disabled_widget_never_activates() {
        for available in [false, true] {
            assert!(
                replay(
                    &[press(10.0, 10.0), lift(10.0, 10.0)],
                    false,
                    available,
                    false
                )
                .0
                .is_empty()
            );
        }
    }
}
