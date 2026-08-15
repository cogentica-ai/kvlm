// flamegraph package: port of Brendan Gregg's FlameGraph.pl to Goish Rust.
//
// Generates SVG flamegraphs from folded stack input (e.g., from
// perf script | stackcollapse-perf.pl).
//
// Folded stack format:
//   func1;func2;func3 count
//   func1;func2;func4 count
//
// The output is an SVG flamegraph showing call stacks with widths
// proportional to sample counts.
#![allow(non_snake_case)]

use goish::fmt;
use goish::strings;
use goish::string;
use goish::goslice::slice;
use goish::errors::error;
use goish::math;
use goish::strconv;
use goish::{make, append, float64, nil, int, range};

// Frame represents a single function frame in the call stack.
#[derive(Clone, Default)]
pub struct Frame {
    pub Name: string,
    pub Count: int,
    pub Children: slice<Frame>,
}

// FlameGraph holds the parsed stack data and rendering configuration.
#[derive(Clone)]
pub struct FlameGraph {
    pub Root: Frame,
    pub Total: int,
    pub Title: string,
    pub Width: int,
    pub Height: int,
    pub FontSize: int,
    // Vertical renders the d3-style zoomable-icicle partition: depth
    // as columns left to right, sibling share as bar HEIGHT. The
    // readable choice for shallow stacks with long names (GPU kernel
    // summaries); the default stays the classic flame.
    pub Vertical: bool,
    pub VHeight: int, // canvas height of the vertical layout
}

// New creates a new FlameGraph with default configuration.
pub fn New() -> FlameGraph {
    FlameGraph {
        Root: Frame::default(),
        Total: 0,
        Title: string("Flame Graph"),
        Width: 1200,
        Height: 16,
        FontSize: 12,
        Vertical: false,
        VHeight: 720,
    }
}

// ParseFolded parses folded stack input and builds the frame tree.
// Input format: "func1;func2;func3 count\n..."
pub fn ParseFolded(input: string) -> (FlameGraph, error) {
    let mut fg = New();
    let lines = strings::Split(input, "\n");
    
    for (_, line) in range!(lines) {
        let line = strings::TrimSpace(line);
        if line == "" {
            continue;
        }
        
        // Split on last space to separate stack from count
        let lastSpace = strings::LastIndex(line.clone(), " ");
        if lastSpace < 0 {
            continue;
        }
        
        let stack = line.slice(0, lastSpace);
        let countStr = line.slice(lastSpace + 1, line.Len());
        
        let (count, err) = strconv::Atoi(countStr);
        if err != nil {
            continue;
        }
        
        // Split stack into frames
        let frames = strings::Split(stack, ";");
        
        // Add to tree
        addToTree(&mut fg.Root, frames, 0, count);
        fg.Total = fg.Total + count;
    }
    
    (fg, nil.into())
}

// addToTree recursively adds frames to the tree.
fn addToTree(node: &mut Frame, frames: slice<string>, idx: int, count: int) {
    if idx >= frames.Len() {
        node.Count = node.Count + count;
        return;
    }
    
    let frameName = frames[idx].clone();
    
    // Find or create child
    let mut childIdx = -1;
    for i in 0..node.Children.Len() {
        if node.Children[i].Name == frameName {
            childIdx = i as int;
            break;
        }
    }
    
    if childIdx < 0 {
        // Create new child
        let child = Frame {
            Name: frameName,
            Count: 0,
            Children: make!([]Frame, 0),
        };
        node.Children = append!(node.Children.clone(), child);
        childIdx = (node.Children.Len() - 1) as int;
    }
    
    // Recurse
    addToTree(&mut node.Children[childIdx as usize], frames, idx + 1, count);
}

// xmlEsc escapes a string for embedding in SVG/XML text and
// attributes; kernel names carry < and > routinely.
fn xmlEsc(s: string) -> string {
    let mut e = strings::ReplaceAll(s, "&", "&amp;");
    e = strings::ReplaceAll(e, "<", "&lt;");
    e = strings::ReplaceAll(e, ">", "&gt;");
    e = strings::ReplaceAll(e, "\"", "&quot;");
    e
}

// inclusive returns a frame's total samples: itself plus everything
// beneath it. Frame widths come from THIS, not Count — Count is the
// leaf self-time of the folded format, and rendering by it was the
// bug that hid every interior frame.
fn inclusive(f: &Frame) -> int {
    let mut t = f.Count;
    for (_, c) in range!(f.Children.clone()) {
        t = t + inclusive(&c);
    }
    t
}

// sortedChildren returns a frame's children ordered by inclusive
// weight, heaviest first — the classic flamegraph taper, and a
// deterministic layout regardless of input order.
fn sortedChildren(f: &Frame) -> alloc::vec::Vec<Frame> {
    let mut v: alloc::vec::Vec<Frame> = alloc::vec::Vec::new();
    for (_, c) in range!(f.Children.clone()) {
        v.push(c.clone());
    }
    v.sort_by(|a, b| inclusive(b).cmp(&inclusive(a)));
    v
}

// maxDepth returns the deepest stack in frames-below-f.
fn maxDepth(f: &Frame) -> int {
    let mut d = 0;
    for (_, c) in range!(f.Children.clone()) {
        let cd = maxDepth(&c) + 1;
        if cd > d {
            d = cd;
        }
    }
    d
}

// warmColor picks a deterministic warm rgb per function name (djb2
// hash), so the same function is the same color across graphs.
fn warmColor(name: string) -> string {
    let mut h: int = 5381;
    let n: &str = name.as_ref();
    for b in n.bytes() {
        h = ((h * 33) + (b as int)) & 0x7fffffff;
    }
    let r = 205 + h % 50;
    let g = (h / 50) % 200;
    let b = (h / 11500) % 55;
    fmt::Sprintf!("rgb(%d,%d,%d)", r, g, b)
}

// frac renders a 0..1 fraction with enough digits for the zoom script.
fn frac(v: float64) -> string {
    fmt::Sprintf!("%v", math::Round(v * 1000000.0) / 1000000.0)
}

// px renders a pixel coordinate to one decimal.
fn px(v: float64) -> string {
    fmt::Sprintf!("%v", math::Round(v * 10.0) / 10.0)
}

// Go: //go:embed flamegraph.js
goish::embed! {
    #[embed("flamegraph.js")]
    static flamegraphJS: string;
}

// geometry shared by ToSVG and renderFrame.
struct geo {
    marginX: float64,
    drawW: float64,
    frameH: int,
    fontSize: int,
    baseY: float64, // y of the bottom (depth-1) row's top edge
    denom: float64,
    // vertical (icicle-partition) mode
    vertical: bool,
    headerH: float64,
    drawH: float64,
    colW: float64,
}

// ToSVG renders a self-contained interactive SVG: flame layout
// (root row at the bottom), one <g> per frame with fractional
// geometry in data attributes, native <title> hover tooltips, and an
// embedded click-to-zoom script. Opened standalone or inside an
// <object> tag the graph is clickable; inside <img> it degrades to a
// static picture.
pub fn ToSVG(fg: &FlameGraph) -> string {
    let frameH = fg.Height;
    let headerH = 40;
    let footerH = 26;
    let marginX = 10.0;
    let denomI = inclusive(&fg.Root);
    let depth = maxDepth(&fg.Root);
    let mut rows = depth;
    if rows < 1 {
        rows = 1;
    }
    let mut svgH = headerH + rows * (frameH + 1) + footerH;
    if fg.Vertical {
        svgH = headerH + fg.VHeight + footerH;
    }
    let drawW = (fg.Width as float64) - 2.0 * marginX;

    let mut orient = "h";
    if fg.Vertical {
        orient = "v";
    }
    let mut b = strings::Builder::new();
    let _ = b.WriteString(fmt::Sprintf!(
        "<svg version=\"1.1\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\" data-orient=\"%s\" data-drawx=\"%s\" data-draww=\"%s\" data-drawy=\"%d\" data-drawh=\"%d\">\n",
        fg.Width,
        svgH,
        fg.Width,
        svgH,
        string(orient),
        px(marginX),
        px(drawW),
        headerH,
        fg.VHeight
    ));
    let _ = b.WriteString(
        "<style>text{font-family:ui-monospace,Menlo,Consolas,monospace;fill:#111}g.f{cursor:pointer}g.f rect{stroke:#14181c;stroke-width:0.4}g.f:hover rect{stroke:#ffffff;stroke-width:1}.chrome{fill:#d7dde3}</style>\n",
    );
    let _ = b.WriteString(fmt::Sprintf!(
        "<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"#14181c\"/>\n",
        fg.Width,
        svgH
    ));
    let _ = b.WriteString(fmt::Sprintf!(
        "<text x=\"%d\" y=\"24\" font-size=\"%d\" class=\"chrome\" text-anchor=\"middle\">%s</text>\n",
        fg.Width / 2,
        fg.FontSize + 4,
        xmlEsc(fg.Title.clone())
    ));
    let _ = b.WriteString(fmt::Sprintf!(
        "<text id=\"kfg-reset\" x=\"%s\" y=\"24\" font-size=\"%d\" class=\"chrome\" style=\"display:none;cursor:pointer;text-decoration:underline\">reset zoom</text>\n",
        px(marginX),
        fg.FontSize
    ));
    let _ = b.WriteString(fmt::Sprintf!(
        "<text id=\"kfg-details\" x=\"%s\" y=\"%d\" font-size=\"%d\" class=\"chrome\"> </text>\n",
        px(marginX),
        svgH - 8,
        fg.FontSize
    ));

    if fg.Total > 0 && denomI > 0 {
        let mut colW = drawW;
        if depth > 0 {
            colW = (drawW - ((depth - 1) as float64)) / (depth as float64);
        }
        let g = geo {
            marginX,
            drawW,
            frameH,
            fontSize: fg.FontSize,
            baseY: ((svgH - footerH - frameH - 1) as float64),
            denom: (denomI as float64),
            vertical: fg.Vertical,
            headerH: (headerH as float64),
            drawH: (fg.VHeight as float64),
            colW,
        };
        let mut xf = 0.0;
        for c in sortedChildren(&fg.Root).iter() {
            let wf = (inclusive(c) as float64) / g.denom;
            renderFrame(&mut b, c, xf, 1, &g);
            xf += wf;
        }
    }

    let _ = b.WriteString("<script><![CDATA[\n");
    let _ = b.WriteString(flamegraphJS.clone());
    let _ = b.WriteString("]]></script>\n");
    let _ = b.WriteString("</svg>\n");
    b.String()
}

// renderFrame emits one frame group and recurses into its children,
// advancing a running fractional x offset by each child's inclusive
// share.
fn renderFrame(b: &mut strings::Builder, f: &Frame, xf: float64, depth: int, g: &geo) {
    let incl = inclusive(f);
    let wf = (incl as float64) / g.denom;
    let mut x = g.marginX + xf * g.drawW;
    let mut w = wf * g.drawW;
    let mut y = g.baseY - ((depth - 1) as float64) * ((g.frameH + 1) as float64);
    let mut h = g.frameH as float64;
    if g.vertical {
        // icicle partition: depth is the column, share is the height
        x = g.marginX + ((depth - 1) as float64) * (g.colW + 1.0);
        w = g.colW;
        y = g.headerH + xf * g.drawH;
        h = wf * g.drawH;
        if h < 0.3 {
            h = 0.3;
        }
    } else if w < 0.2 {
        w = 0.2;
    }
    let pct = math::Round(wf * 10000.0) / 100.0;
    let name = xmlEsc(f.Name.clone());

    let _ = b.WriteString(fmt::Sprintf!(
        "<g class=\"f\" data-x=\"%s\" data-w=\"%s\" data-d=\"%d\" data-n=\"%s\" data-s=\"%d\" data-p=\"%v\">\n",
        frac(xf),
        frac(wf),
        depth,
        name.clone(),
        incl,
        pct
    ));
    let _ = b.WriteString(fmt::Sprintf!(
        "<title>%s (%d samples, %v%%)</title>\n",
        name.clone(),
        incl,
        pct
    ));
    let _ = b.WriteString(fmt::Sprintf!(
        "<rect x=\"%s\" y=\"%s\" width=\"%s\" height=\"%s\" fill=\"%s\" rx=\"1\"/>\n",
        px(x),
        px(y),
        px(w),
        px(h),
        warmColor(f.Name.clone())
    ));
    let mut showText = w > 40.0 && !g.vertical;
    if g.vertical {
        showText = h >= ((g.fontSize + 4) as float64);
    }
    if showText {
        let maxChars = (((w - 6.0) / 7.0) as int);
        let display = truncateString(f.Name.clone(), maxChars);
        let mut textY = y + ((g.frameH as float64) - 4.0);
        if g.vertical {
            textY = y + ((g.fontSize as float64)) + 2.0;
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "<text x=\"%s\" y=\"%s\" font-size=\"%d\">%s</text>\n",
            px(x + 3.0),
            px(textY),
            g.fontSize,
            xmlEsc(display)
        ));
    }
    let _ = b.WriteString("</g>\n");

    let mut cx = xf;
    for c in sortedChildren(f).iter() {
        let cwf = (inclusive(c) as float64) / g.denom;
        renderFrame(b, c, cx, depth + 1, g);
        cx += cwf;
    }
}

// truncateString truncates a string to maxLen characters and adds "..." if needed.
fn truncateString(s: string, maxLen: int) -> string {
    if s.Len() <= maxLen {
        return s;
    }
    if maxLen <= 3 {
        return s.slice(0, maxLen);
    }
    return (s.slice(0, maxLen - 3)) + ("...");
}
