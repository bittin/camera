// SPDX-License-Identifier: GPL-3.0-only

//! Headless real-widget helpers shared by UI characterization tests.

use crate::app::state::Message;
use cosmic::Element;
use cosmic::iced::advanced::{
    Layout, Shell, clipboard, layout,
    renderer::Headless,
    widget::{Id, Operation, Tree},
};
use cosmic::iced::{Event, Font, Pixels, Point, Rectangle, Size, mouse};

pub(crate) fn renderer() -> cosmic::Renderer {
    pollster::block_on(cosmic::Renderer::new(
        Font::DEFAULT,
        Pixels(14.0),
        Some("tiny-skia"),
    ))
    .expect("headless software renderer for UI characterization")
}

pub(crate) fn layout(
    element: &mut Element<'_, Message>,
    renderer: &cosmic::Renderer,
    size: Size,
) -> (Tree, layout::Node) {
    let mut tree = Tree::new(element.as_widget());
    element.as_widget_mut().diff(&mut tree);
    let node =
        element
            .as_widget_mut()
            .layout(&mut tree, renderer, &layout::Limits::new(Size::ZERO, size));
    (tree, node)
}

#[derive(Default)]
struct Texts(Vec<(String, Rectangle)>);
impl Operation for Texts {
    fn traverse(&mut self, visit: &mut dyn FnMut(&mut dyn Operation)) {
        visit(self);
    }
    fn text(&mut self, _: Option<&Id>, bounds: Rectangle, text: &str) {
        self.0.push((text.to_owned(), bounds));
    }
}

pub(crate) fn texts(
    element: &mut Element<'_, Message>,
    tree: &mut Tree,
    node: &layout::Node,
    renderer: &cosmic::Renderer,
) -> Vec<(String, Rectangle)> {
    let mut texts = Texts::default();
    element
        .as_widget_mut()
        .operate(tree, Layout::new(node), renderer, &mut texts);
    texts.0
}

pub(crate) fn send(
    element: &mut Element<'_, Message>,
    tree: &mut Tree,
    node: &layout::Node,
    renderer: &cosmic::Renderer,
    event: Event,
    cursor: mouse::Cursor,
) -> Vec<Message> {
    let mut messages = Vec::new();
    element.as_widget_mut().update(
        tree,
        &event,
        Layout::new(node),
        cursor,
        renderer,
        &mut clipboard::Null,
        &mut Shell::new(&mut messages),
        &Rectangle::with_size(Size::new(1200.0, 2400.0)),
    );
    messages
}

pub(crate) fn click(
    element: &mut Element<'_, Message>,
    tree: &mut Tree,
    node: &layout::Node,
    renderer: &cosmic::Renderer,
    position: Point,
) -> Vec<Message> {
    [
        mouse::Event::ButtonPressed(mouse::Button::Left),
        mouse::Event::ButtonReleased(mouse::Button::Left),
    ]
    .into_iter()
    .flat_map(|event| {
        send(
            element,
            tree,
            node,
            renderer,
            Event::Mouse(event),
            mouse::Cursor::Available(position),
        )
    })
    .collect()
}
