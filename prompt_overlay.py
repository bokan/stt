"""Prompt selection overlay with keyboard shortcuts"""

import objc
import threading
from typing import Callable, Optional

from AppKit import (
    NSWindow,
    NSWindowStyleMaskBorderless,
    NSBackingStoreBuffered,
    NSFloatingWindowLevel,
    NSScreen,
    NSView,
    NSColor,
    NSBezierPath,
    NSEvent,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSTrackingArea,
    NSTrackingMouseEnteredAndExited,
    NSTrackingActiveAlways,
    NSTrackingInVisibleRect,
)
from Foundation import (
    NSRect,
    NSPoint,
    NSMakeRect,
    NSPointInRect,
    NSDictionary,
    NSAttributedString,
)

from prompts_config import PromptItem

# Overlay dimensions
OVERLAY_WIDTH = 320
ITEM_HEIGHT = 44
PADDING = 12
CORNER_RADIUS = 12
BADGE_PADDING = 6
BADGE_CORNER = 4


class PromptOverlayView(NSView):
    """Custom view that draws prompt list with hover highlight"""

    def initWithFrame_prompts_callback_(self, frame, prompts, callback):
        self = objc.super(PromptOverlayView, self).initWithFrame_(frame)
        if self:
            self._prompts = prompts
            self._callback = callback
            self._hovered_index = -1
            self._setup_tracking()
        return self

    def _setup_tracking(self):
        """Setup mouse tracking for hover effects"""
        options = (
            NSTrackingMouseEnteredAndExited |
            NSTrackingActiveAlways |
            NSTrackingInVisibleRect
        )
        area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            self.bounds(), options, self, None
        )
        self.addTrackingArea_(area)

    def updatePrompts_(self, prompts):
        """Update prompt list and redraw"""
        self._prompts = prompts
        self._hovered_index = -1
        self.setNeedsDisplay_(True)

    def _item_rect(self, index):
        """Get rect for item at index"""
        bounds = self.bounds()
        y = bounds.size.height - PADDING - (index + 1) * ITEM_HEIGHT
        return NSMakeRect(PADDING, y, bounds.size.width - 2 * PADDING, ITEM_HEIGHT)

    def _index_at_point(self, point):
        """Get item index at point, or -1"""
        for i in range(len(self._prompts)):
            if NSPointInRect(point, self._item_rect(i)):
                return i
        return -1

    def mouseMoved_(self, event):
        """Track mouse movement for hover"""
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        new_index = self._index_at_point(loc)
        if new_index != self._hovered_index:
            self._hovered_index = new_index
            self.setNeedsDisplay_(True)

    def mouseEntered_(self, event):
        """Mouse entered view"""
        self.mouseMoved_(event)

    def mouseExited_(self, event):
        """Mouse exited view"""
        if self._hovered_index != -1:
            self._hovered_index = -1
            self.setNeedsDisplay_(True)

    def mouseDown_(self, event):
        """Handle click on item"""
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        index = self._index_at_point(loc)
        if 0 <= index < len(self._prompts):
            prompt = self._prompts[index]
            if self._callback:
                self._callback(prompt.text, prompt.enter)

    def drawRect_(self, rect):
        """Draw background and prompt items"""
        bounds = self.bounds()

        # Dark semi-transparent background
        bg_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.85)
        bg_color.setFill()
        bg_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, CORNER_RADIUS, CORNER_RADIUS
        )
        bg_path.fill()

        # Draw each prompt item
        for i, prompt in enumerate(self._prompts):
            item_rect = self._item_rect(i)
            self._draw_item(prompt, item_rect, i == self._hovered_index)

    def _draw_item(self, prompt, rect, hovered):
        """Draw a single prompt item"""
        # Hover highlight
        if hovered:
            hover_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.1)
            hover_color.setFill()
            hover_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                rect, 6, 6
            )
            hover_path.fill()

        # Icon (emoji)
        icon_font = NSFont.systemFontOfSize_(20)
        icon_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
            [icon_font, NSColor.whiteColor()],
            [NSFontAttributeName, NSForegroundColorAttributeName]
        )
        icon_str = NSAttributedString.alloc().initWithString_attributes_(
            prompt.icon, icon_attrs
        )
        icon_x = rect.origin.x + 8
        icon_y = rect.origin.y + (rect.size.height - 24) / 2
        icon_str.drawAtPoint_(NSPoint(icon_x, icon_y))

        # Label
        label_font = NSFont.systemFontOfSize_(14)
        label_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
            [label_font, NSColor.whiteColor()],
            [NSFontAttributeName, NSForegroundColorAttributeName]
        )
        label_str = NSAttributedString.alloc().initWithString_attributes_(
            prompt.label, label_attrs
        )
        label_x = icon_x + 36
        label_y = rect.origin.y + (rect.size.height - 18) / 2
        label_str.drawAtPoint_(NSPoint(label_x, label_y))

        # Key badge on right
        if prompt.key:
            badge_font = NSFont.monospacedSystemFontOfSize_weight_(11, 0.5)
            badge_text = f"[{prompt.key}]"
            badge_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                [badge_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.6)],
                [NSFontAttributeName, NSForegroundColorAttributeName]
            )
            badge_str = NSAttributedString.alloc().initWithString_attributes_(
                badge_text, badge_attrs
            )
            badge_size = badge_str.size()

            # Badge background
            badge_x = rect.origin.x + rect.size.width - badge_size.width - BADGE_PADDING - 8
            badge_y = rect.origin.y + (rect.size.height - badge_size.height) / 2
            badge_rect = NSMakeRect(
                badge_x - BADGE_PADDING,
                badge_y - 2,
                badge_size.width + 2 * BADGE_PADDING,
                badge_size.height + 4
            )
            badge_bg = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.1)
            badge_bg.setFill()
            badge_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                badge_rect, BADGE_CORNER, BADGE_CORNER
            )
            badge_path.fill()

            badge_str.drawAtPoint_(NSPoint(badge_x, badge_y))


class PromptOverlay:
    """Manages the prompt selection overlay window"""

    def __init__(self, on_select: Callable[[str], None]):
        self._on_select = on_select
        self._window: Optional[NSWindow] = None
        self._view: Optional[PromptOverlayView] = None
        self._visible = False
        self._lock = threading.Lock()
        # Load prompts synchronously at init
        from prompts_config import load_prompts
        self._prompts: list[PromptItem] = load_prompts()

    def _ensure_window(self):
        """Create window if not exists (must be called on main thread)"""
        if self._window is not None:
            return

        height = self._calculate_height()
        frame = self._calculate_frame(height)

        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )

        window.setLevel_(NSFloatingWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setIgnoresMouseEvents_(False)
        window.setHasShadow_(True)
        window.setAcceptsMouseMovedEvents_(True)

        view_frame = NSMakeRect(0, 0, OVERLAY_WIDTH, height)
        view = PromptOverlayView.alloc().initWithFrame_prompts_callback_(
            view_frame, self._prompts, self._handle_selection
        )
        window.setContentView_(view)

        self._window = window
        self._view = view

    def _calculate_height(self):
        """Calculate overlay height based on prompt count"""
        item_count = max(1, len(self._prompts))
        return 2 * PADDING + item_count * ITEM_HEIGHT

    def _calculate_frame(self, height):
        """Calculate window frame centered on mouse screen"""
        mouse_loc = NSEvent.mouseLocation()
        screen = None
        for s in NSScreen.screens():
            if NSPointInRect(mouse_loc, s.frame()):
                screen = s
                break
        if screen is None:
            screen = NSScreen.mainScreen()
        screen_frame = screen.frame()

        x = screen_frame.origin.x + (screen_frame.size.width - OVERLAY_WIDTH) / 2
        y = screen_frame.origin.y + (screen_frame.size.height - height) / 2

        return NSMakeRect(x, y, OVERLAY_WIDTH, height)

    def _handle_selection(self, text: str, send_enter: bool = False):
        """Called when prompt selected"""
        self.hide()
        if self._on_select:
            self._on_select(text, send_enter)

    def reload_prompts(self):
        """Reload prompts from config"""
        from prompts_config import load_prompts
        self._prompts = load_prompts()

        def _update():
            if self._view:
                self._view.updatePrompts_(self._prompts)
                # Resize window
                height = self._calculate_height()
                if self._window:
                    frame = self._calculate_frame(height)
                    self._window.setFrame_display_(frame, True)
                    self._view.setFrame_(NSMakeRect(0, 0, OVERLAY_WIDTH, height))

        _run_on_main_thread(_update)

    def show(self):
        """Show the overlay"""
        with self._lock:
            if self._visible:
                return
            self._visible = True

        def _show():
            self.reload_prompts()
            self._ensure_window()
            height = self._calculate_height()
            frame = self._calculate_frame(height)
            self._window.setFrame_display_(frame, True)
            self._view.setFrame_(NSMakeRect(0, 0, OVERLAY_WIDTH, height))
            self._window.orderFront_(None)

        _run_on_main_thread(_show)

    def hide(self):
        """Hide the overlay"""
        with self._lock:
            if not self._visible:
                return
            self._visible = False

        def _hide():
            if self._window:
                self._window.orderOut_(None)

        _run_on_main_thread(_hide)

    def handle_key(self, char: str) -> bool:
        """Check if key matches a prompt shortcut. Returns True if matched."""
        with self._lock:
            if not self._visible:
                return False

        for prompt in self._prompts:
            # Check enter flag for enter key
            if char == "enter" and prompt.enter:
                self._handle_selection(prompt.text, send_enter=True)
                return True
            # Check key match
            if prompt.key and prompt.key.lower() == char.lower():
                self._handle_selection(prompt.text, send_enter=prompt.enter)
                return True
        return False


def _run_on_main_thread(func):
    """Run function on main thread"""
    from Foundation import NSThread, NSRunLoop, NSRunLoopCommonModes

    if NSThread.isMainThread():
        func()
        return

    from AppKit import NSApplication
    from Foundation import NSTimer

    app = NSApplication.sharedApplication()
    if app:
        def fire_(_):
            func()

        timer = NSTimer.timerWithTimeInterval_repeats_block_(0, False, fire_)
        NSRunLoop.mainRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
