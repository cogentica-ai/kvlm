// Test cases for the flamegraph module.
// Tests cover: parsing folded stacks, tree building, SVG rendering, edge cases.
#![no_std]
#![no_main]

extern crate alloc;

use goish::fmt;
use goish::string;
use goish::testing;
use kvlm::flamegraph;

// TestToSVG_ZeroTotal: an empty graph (no folded lines parsed) must
// render a valid empty SVG rather than divide by zero.
fn TestToSVG_ZeroTotal(t: &mut testing::T) {
    let fg = flamegraph::New();
    if fg.Total != 0 {
        t.Fatal(fmt::Sprintf!("fresh graph total: got %d, want 0", fg.Total));
    }
    let svg = flamegraph::ToSVG(&fg);
    if svg.Len() == 0 {
        t.Fatal(string("empty graph must still produce SVG scaffolding"));
    }
}

// TestToSVG_InclusiveWidths: interior frames must render with their
// inclusive (self + descendants) share — the old renderer credited
// only leaf counts and dropped every interior frame.
fn TestToSVG_InclusiveWidths(t: &mut testing::T) {
    let (fg, _) = flamegraph::ParseFolded(string("main;foo;bar 10\nmain;foo;baz 5\n"));
    let svg = flamegraph::ToSVG(&fg);
    if !goish::strings::Contains(svg.clone(), "data-n=\"main\"") {
        t.Fatal(string("interior frame main missing from SVG"));
    }
    if !goish::strings::Contains(svg.clone(), "data-w=\"1\"") {
        t.Fatal(string("main should span the full width (data-w 1)"));
    }
    if !goish::strings::Contains(svg.clone(), "main (15 samples, 100%)") {
        t.Fatal(string("main tooltip should carry inclusive samples and percent"));
    }
}

// TestToSVG_Interactive: the SVG must be self-contained interactive.
fn TestToSVG_Interactive(t: &mut testing::T) {
    let (fg, _) = flamegraph::ParseFolded(string("a;b 1\n"));
    let svg = flamegraph::ToSVG(&fg);
    if !goish::strings::Contains(svg.clone(), "<script>") {
        t.Fatal(string("zoom script missing"));
    }
    if !goish::strings::Contains(svg.clone(), "data-x=") {
        t.Fatal(string("fractional geometry attributes missing"));
    }
    if !goish::strings::Contains(svg.clone(), "reset zoom") {
        t.Fatal(string("reset control missing"));
    }
}

// TestToSVG_EscapesXML: kernel names carry < > & routinely; they must
// never reach the SVG raw.
fn TestToSVG_EscapesXML(t: &mut testing::T) {
    let (fg, _) = flamegraph::ParseFolded(string("void cutlass::device_kernel<flash::Fwd&Combine> 3\n"));
    let svg = flamegraph::ToSVG(&fg);
    if goish::strings::Contains(svg.clone(), "device_kernel<flash") {
        t.Fatal(string("raw < leaked into the SVG"));
    }
    if !goish::strings::Contains(svg.clone(), "device_kernel&lt;flash::Fwd&amp;Combine&gt;") {
        t.Fatal(string("name not escaped as expected"));
    }
}

// TestToSVG_DepthHeight: taller stacks need a taller SVG.
fn TestToSVG_DepthHeight(t: &mut testing::T) {
    let (shallow, _) = flamegraph::ParseFolded(string("a 1\n"));
    let (deep, _) = flamegraph::ParseFolded(string("a;b;c;d;e;f 1\n"));
    let sSvg = flamegraph::ToSVG(&shallow);
    let dSvg = flamegraph::ToSVG(&deep);
    // shallow: 1 row; deep: 6 rows. Compare the height attributes.
    if !goish::strings::Contains(sSvg.clone(), "height=\"83\"") {
        t.Fatal(fmt::Sprintf!("shallow height drifted: %s", sSvg.slice(0, 120)));
    }
    if !goish::strings::Contains(dSvg.clone(), "height=\"168\"") {
        t.Fatal(fmt::Sprintf!("deep height drifted: %s", dSvg.slice(0, 120)));
    }
}

// TestToSVG_Vertical: the partition layout stacks share on the y axis
// and marks the orientation for the zoom script.
fn TestToSVG_Vertical(t: &mut testing::T) {
    let (fg, _) = flamegraph::ParseFolded(string("main;foo;bar 10\nmain;foo;baz 5\n"));
    let mut fg = fg;
    fg.Vertical = true;
    let svg = flamegraph::ToSVG(&fg);
    if !goish::strings::Contains(svg.clone(), "data-orient=\"v\"") {
        t.Fatal(string("vertical orientation marker missing"));
    }
    // canvas height: 40 header + 720 partition + 26 footer
    if !goish::strings::Contains(svg.clone(), "height=\"786\"") {
        t.Fatal(string("vertical canvas height drifted"));
    }
    // main spans the full partition: a rect of height 720
    if !goish::strings::Contains(svg.clone(), "height=\"720\" fill=") {
        t.Fatal(string("root frame should span the full partition height"));
    }
    // horizontal default unchanged
    let (fg2, _) = flamegraph::ParseFolded(string("a;b 1\n"));
    let svg2 = flamegraph::ToSVG(&fg2);
    if !goish::strings::Contains(svg2.clone(), "data-orient=\"h\"") {
        t.Fatal(string("horizontal default lost its orientation marker"));
    }
}

// TestParseFolded_Basic tests basic folded stack parsing.
fn TestParseFolded_Basic(t: &mut testing::T) {
    let input = string("main;foo;bar 10\nmain;foo;baz 5\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    if fg.Total != 15 {
        t.Fatal(fmt::Sprintf!("expected total 15, got %d", fg.Total));
    }
    
    // Check root has one child (main)
    if fg.Root.Children.Len() != 1 {
        t.Fatal(fmt::Sprintf!("expected 1 root child, got %d", fg.Root.Children.Len()));
    }
    
    let main_frame = &fg.Root.Children[0];
    if main_frame.Name != "main" {
        t.Fatal(fmt::Sprintf!("expected root child 'main', got '%s'", main_frame.Name.clone()));
    }
    
    // Check main has one child (foo)
    if main_frame.Children.Len() != 1 {
        t.Fatal(fmt::Sprintf!("expected 1 main child, got %d", main_frame.Children.Len()));
    }
    
    let foo_frame = &main_frame.Children[0];
    if foo_frame.Name != "foo" {
        t.Fatal(fmt::Sprintf!("expected main child 'foo', got '%s'", foo_frame.Name.clone()));
    }
    
    // Check foo has two children (bar and baz)
    if foo_frame.Children.Len() != 2 {
        t.Fatal(fmt::Sprintf!("expected 2 foo children, got %d", foo_frame.Children.Len()));
    }
}

// TestParseFolded_EmptyInput tests parsing empty input.
fn TestParseFolded_EmptyInput(t: &mut testing::T) {
    let input = string("");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed on empty input: %v", err));
    }
    
    if fg.Total != 0 {
        t.Fatal(fmt::Sprintf!("expected total 0 for empty input, got %d", fg.Total));
    }
    
    if fg.Root.Children.Len() != 0 {
        t.Fatal(fmt::Sprintf!("expected 0 root children for empty input, got %d", fg.Root.Children.Len()));
    }
}

// TestParseFolded_InvalidCount tests parsing with invalid count values.
fn TestParseFolded_InvalidCount(t: &mut testing::T) {
    let input = string("main;foo abc\nmain;bar 10\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    // Should skip the invalid line and only count the valid one
    if fg.Total != 10 {
        t.Fatal(fmt::Sprintf!("expected total 10 (skipping invalid), got %d", fg.Total));
    }
}

// TestParseFolded_NoSpace tests parsing lines without space separator.
fn TestParseFolded_NoSpace(t: &mut testing::T) {
    let input = string("main;foo\nmain;bar 5\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    // Should skip the line without space
    if fg.Total != 5 {
        t.Fatal(fmt::Sprintf!("expected total 5 (skipping no-space line), got %d", fg.Total));
    }
}

// TestParseFolded_SingleFrame tests parsing single-frame stacks.
fn TestParseFolded_SingleFrame(t: &mut testing::T) {
    let input = string("main 100\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    if fg.Total != 100 {
        t.Fatal(fmt::Sprintf!("expected total 100, got %d", fg.Total));
    }
    
    if fg.Root.Children.Len() != 1 {
        t.Fatal(fmt::Sprintf!("expected 1 root child, got %d", fg.Root.Children.Len()));
    }
    
    if fg.Root.Children[0].Name != "main" {
        t.Fatal(fmt::Sprintf!("expected 'main', got '%s'", fg.Root.Children[0].Name.clone()));
    }
}

// TestParseFolded_DeepStack tests parsing deeply nested stacks.
fn TestParseFolded_DeepStack(t: &mut testing::T) {
    let input = string("a;b;c;d;e;f;g;h;i;j 1\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    if fg.Total != 1 {
        t.Fatal(fmt::Sprintf!("expected total 1, got %d", fg.Total));
    }
    
    // Walk down the tree to verify depth
    let mut current = &fg.Root;
    let expected = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
    
    for i in 0..10 {
        if current.Children.Len() != 1 {
            t.Fatal(fmt::Sprintf!("expected 1 child at depth %d, got %d", i, current.Children.Len()));
        }
        current = &current.Children[0];
        if current.Name != expected[i] {
            t.Fatal(fmt::Sprintf!("expected '%s' at depth %d, got '%s'", expected[i], i, current.Name.clone()));
        }
    }
}

// TestParseFolded_MergingStacks tests that identical stacks are merged.
fn TestParseFolded_MergingStacks(t: &mut testing::T) {
    let input = string("main;foo 5\nmain;foo 3\nmain;foo 2\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    if fg.Total != 10 {
        t.Fatal(fmt::Sprintf!("expected total 10, got %d", fg.Total));
    }
    
    // Should have merged into a single path
    if fg.Root.Children.Len() != 1 {
        t.Fatal(fmt::Sprintf!("expected 1 root child, got %d", fg.Root.Children.Len()));
    }
    
    let main_frame = &fg.Root.Children[0];
    if main_frame.Children.Len() != 1 {
        t.Fatal(fmt::Sprintf!("expected 1 main child, got %d", main_frame.Children.Len()));
    }
    
    let foo_frame = &main_frame.Children[0];
    if foo_frame.Count != 10 {
        t.Fatal(fmt::Sprintf!("expected foo count 10, got %d", foo_frame.Count));
    }
}

// TestToSVG_Basic tests basic SVG generation.
fn TestToSVG_Basic(t: &mut testing::T) {
    let input = string("main;foo 10\nmain;bar 5\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    let svg = flamegraph::ToSVG(&fg);
    
    // Check SVG contains expected elements
    if !goish::strings::Contains(svg.clone(), "<svg") {
        t.Fatal("SVG missing <svg> tag");
    }
    
    if !goish::strings::Contains(svg.clone(), "</svg>") {
        t.Fatal("SVG missing </svg> tag");
    }
    
    if !goish::strings::Contains(svg.clone(), "<rect") {
        t.Fatal("SVG missing <rect> elements");
    }
    
    if !goish::strings::Contains(svg.clone(), "<text") {
        t.Fatal("SVG missing <text> elements");
    }
    
    if !goish::strings::Contains(svg.clone(), "Flame Graph") {
        t.Fatal("SVG missing title");
    }
}

// TestToSVG_Empty tests SVG generation for empty flamegraph.
fn TestToSVG_Empty(t: &mut testing::T) {
    let fg = flamegraph::New();
    let svg = flamegraph::ToSVG(&fg);
    
    // Should still have valid SVG structure
    if !goish::strings::Contains(svg.clone(), "<svg") {
        t.Fatal("Empty SVG missing <svg> tag");
    }
    
    if !goish::strings::Contains(svg.clone(), "</svg>") {
        t.Fatal("Empty SVG missing </svg> tag");
    }
}

// TestToSVG_CustomTitle tests SVG generation with custom title.
fn TestToSVG_CustomTitle(t: &mut testing::T) {
    let mut fg = flamegraph::New();
    fg.Title = string("Custom Title");
    
    let input = string("main 10\n");
    let (parsed, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    fg.Root = parsed.Root;
    fg.Total = parsed.Total;
    
    let svg = flamegraph::ToSVG(&fg);
    
    if !goish::strings::Contains(svg.clone(), "Custom Title") {
        t.Fatal("SVG missing custom title");
    }
}

// TestTruncateString tests the string truncation helper.
fn TestTruncateString(t: &mut testing::T) {
    // Note: truncateString is not exported, so we test it indirectly through ToSVG
    // by creating frames with long names
    
    let input = string("very_long_function_name_that_exceeds_normal_width 10\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    let svg = flamegraph::ToSVG(&fg);
    
    // The long name should be truncated in the SVG
    // We can't directly test truncateString, but we can verify the SVG is valid
    if !goish::strings::Contains(svg.clone(), "<svg") {
        t.Fatal("SVG with long name is invalid");
    }
}

// TestParseFolded_ZeroCount tests parsing with zero count.
fn TestParseFolded_ZeroCount(t: &mut testing::T) {
    let input = string("main;foo 0\nmain;bar 5\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    // Zero count should still be parsed
    if fg.Total != 5 {
        t.Fatal(fmt::Sprintf!("expected total 5, got %d", fg.Total));
    }
}

// TestParseFolded_LargeCount tests parsing with large count values.
fn TestParseFolded_LargeCount(t: &mut testing::T) {
    let input = string("main;foo 999999\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    if fg.Total != 999999 {
        t.Fatal(fmt::Sprintf!("expected total 999999, got %d", fg.Total));
    }
}

// TestParseFolded_Whitespace tests parsing with extra whitespace.
fn TestParseFolded_Whitespace(t: &mut testing::T) {
    let input = string("  main;foo  10  \n\nmain;bar 5\n");
    let (fg, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    // Should handle whitespace gracefully
    if fg.Total != 15 {
        t.Fatal(fmt::Sprintf!("expected total 15, got %d", fg.Total));
    }
}

// TestToSVG_Dimensions tests that SVG respects custom dimensions.
fn TestToSVG_Dimensions(t: &mut testing::T) {
    let mut fg = flamegraph::New();
    fg.Width = 800;
    fg.Height = 20;
    
    let input = string("main 10\n");
    let (parsed, err) = flamegraph::ParseFolded(input);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("ParseFolded failed: %v", err));
    }
    
    fg.Root = parsed.Root;
    fg.Total = parsed.Total;
    
    let svg = flamegraph::ToSVG(&fg);
    
    // Check that custom dimensions are in the SVG
    if !goish::strings::Contains(svg.clone(), "width=\"800\"") {
        t.Fatal("SVG missing custom width");
    }
}

// Main test runner.
#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestToSVG_ZeroTotal", TestToSVG_ZeroTotal),
        ("TestToSVG_InclusiveWidths", TestToSVG_InclusiveWidths),
        ("TestToSVG_Interactive", TestToSVG_Interactive),
        ("TestToSVG_EscapesXML", TestToSVG_EscapesXML),
        ("TestToSVG_DepthHeight", TestToSVG_DepthHeight),
        ("TestToSVG_Vertical", TestToSVG_Vertical),
        ("TestParseFolded_Basic", TestParseFolded_Basic),
        ("TestParseFolded_EmptyInput", TestParseFolded_EmptyInput),
        ("TestParseFolded_InvalidCount", TestParseFolded_InvalidCount),
        ("TestParseFolded_NoSpace", TestParseFolded_NoSpace),
        ("TestParseFolded_SingleFrame", TestParseFolded_SingleFrame),
        ("TestParseFolded_DeepStack", TestParseFolded_DeepStack),
        ("TestParseFolded_MergingStacks", TestParseFolded_MergingStacks),
        ("TestToSVG_Basic", TestToSVG_Basic),
        ("TestToSVG_Empty", TestToSVG_Empty),
        ("TestToSVG_CustomTitle", TestToSVG_CustomTitle),
        ("TestTruncateString", TestTruncateString),
        ("TestParseFolded_ZeroCount", TestParseFolded_ZeroCount),
        ("TestParseFolded_LargeCount", TestParseFolded_LargeCount),
        ("TestParseFolded_Whitespace", TestParseFolded_Whitespace),
        ("TestToSVG_Dimensions", TestToSVG_Dimensions),
    ];
    let code = testing::Main(tests);
    goish::syscall::Exit(goish::int32(code));
}
