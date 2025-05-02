// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded "><a href="howto.html">🧩 Puzzles Usage Guide</a></li><li class="chapter-item expanded affix "><li class="part-title">Part I: GPU Fundamentals</li><li class="chapter-item expanded "><a href="puzzle_01/puzzle_01.html">Puzzle 1: Map</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_01/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_01/layout_tensor_preview.html">💡 Preview: Modern Approach with LayoutTensor</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_02/puzzle_02.html">Puzzle 2: Zip</a></li><li class="chapter-item expanded "><a href="puzzle_03/puzzle_03.html">Puzzle 3: Guards</a></li><li class="chapter-item expanded "><a href="puzzle_04/puzzle_04.html">Puzzle 4: 2D Map</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_04/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_04/introduction_layout_tensor.html">📚 Learn about LayoutTensor</a></li><li class="chapter-item expanded "><a href="puzzle_04/layout_tensor.html">🚀 Modern 2D Operations</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_05/puzzle_05.html">Puzzle 5: Broadcast</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_05/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_05/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_06/puzzle_06.html">Puzzle 6: Blocks</a></li><li class="chapter-item expanded "><a href="puzzle_07/puzzle_07.html">Puzzle 7: 2D Blocks</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_07/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_07/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_08/puzzle_08.html">Puzzle 8: Shared Memory</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_08/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_08/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part II: GPU Algorithms</li><li class="chapter-item expanded "><a href="puzzle_09/puzzle_09.html">Puzzle 9: Pooling</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_09/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_09/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_10/puzzle_10.html">Puzzle 10: Dot Product</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_10/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_10/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_11/puzzle_11.html">Puzzle 11: 1D Convolution</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_11/simple.html">🔰 Simple Version</a></li><li class="chapter-item expanded "><a href="puzzle_11/complete.html">⭐ Complete Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_12/puzzle_12.html">Puzzle 12: Prefix Sum</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_12/simple.html">🔰 Simple Version</a></li><li class="chapter-item expanded "><a href="puzzle_12/complete.html">⭐ Complete Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_13/puzzle_13.html">Puzzle 13: Axis Sum</a></li><li class="chapter-item expanded "><a href="puzzle_14/puzzle_14.html">Puzzle 14: Matrix Multiplication (MatMul)</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_14/naive.html">🔰 Naive Version with Global Memory</a></li><li class="chapter-item expanded "><div>📚 Learn about Roofline Model</div></li><li class="chapter-item expanded "><a href="puzzle_14/shared_memory.html">🤝 Shared Memory Version</a></li><li class="chapter-item expanded "><a href="puzzle_14/tiled.html">📐 Tiled Version</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part III: Interfacing with Python via MAX Graph Custom Ops</li><li class="chapter-item expanded "><div>Puzzle 15: 1D Convolution Op</div></li><li class="chapter-item expanded "><div>Puzzle 16: Softmax Op</div></li><li class="chapter-item expanded "><div>Puzzle 17: Attention Op</div></li><li class="chapter-item expanded "><div>🎯 Bonus Challenges</div></li><li class="chapter-item expanded affix "><li class="part-title">Part IV: Advanced GPU Algorithms</li><li class="chapter-item expanded "><div>Puzzle 18: 2D Convolution Op</div></li><li class="chapter-item expanded "><div>Puzzle 19: 3D Average Pooling</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about 3D Memory Layout</div></li><li class="chapter-item expanded "><div>Basic Version</div></li><li class="chapter-item expanded "><div>LayoutTensor Version</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 20: 3D Convolution</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about 3D Convolution</div></li><li class="chapter-item expanded "><div>Basic Version</div></li><li class="chapter-item expanded "><div>Optimized Version</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 21: 3D Tensor Multiplication</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Tensor Operations</div></li><li class="chapter-item expanded "><div>Basic Version</div></li><li class="chapter-item expanded "><div>LayoutTensor Version</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 22: Multi-Head Self-Attention</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Attention Mechanisms</div></li><li class="chapter-item expanded "><div>Basic Version</div></li><li class="chapter-item expanded "><div>Optimized Version</div></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part V: Performance Optimization Puzzles</li><li class="chapter-item expanded "><div>Puzzle 23: Memory Coalescing</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Memory Access Patterns</div></li><li class="chapter-item expanded "><div>Basic Version</div></li><li class="chapter-item expanded "><div>Optimized Version</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 24: Bank Conflicts</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Shared Memory Banks</div></li><li class="chapter-item expanded "><div>Version 1: With Conflicts</div></li><li class="chapter-item expanded "><div>Version 2: Conflict-Free</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 25: Warp-Level Optimization</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Warp Primitives</div></li><li class="chapter-item expanded "><div>Version 1: Shared Memory Reduction</div></li><li class="chapter-item expanded "><div>Version 2: Warp Shuffle Reduction</div></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part VI: Real-world Application Puzzles</li><li class="chapter-item expanded "><div>Puzzle 26: Image Processing Pipeline</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Kernel Fusion</div></li><li class="chapter-item expanded "><div>Version 1: Separate Kernels</div></li><li class="chapter-item expanded "><div>Version 2: Fused Pipeline</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 27: Neural Network Layers</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Layer Fusion</div></li><li class="chapter-item expanded "><div>Version 1: Basic Implementation</div></li><li class="chapter-item expanded "><div>Version 2: Optimized Implementation</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 28: Multi-Level Tiling</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Cache Hierarchies</div></li><li class="chapter-item expanded "><div>Version 1: Single-Level MatMul</div></li><li class="chapter-item expanded "><div>Version 2: Multi-Level MatMul</div></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part VII: Debug &amp; Profile Puzzles</li><li class="chapter-item expanded "><div>Puzzle 29: Race Condition Detective</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Race Conditions</div></li><li class="chapter-item expanded "><div>Version 1: Find the Bug</div></li><li class="chapter-item expanded "><div>Version 2: Fix the Bug</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 30: Memory Optimization</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Memory Management</div></li><li class="chapter-item expanded "><div>Version 1: Memory Leaks</div></li><li class="chapter-item expanded "><div>Version 2: Memory Planning</div></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part VIII: Modern GPU Features</li><li class="chapter-item expanded "><div>Puzzle 31: Dynamic Parallelism</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Nested Parallelism</div></li><li class="chapter-item expanded "><div>Version 1: Flat Implementation</div></li><li class="chapter-item expanded "><div>Version 2: Nested Launch</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 32: Tensor Core Programming</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Tensor Cores</div></li><li class="chapter-item expanded "><div>Version 1: Regular MatMul</div></li><li class="chapter-item expanded "><div>Version 2: Tensor Core MatMul</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 33: Multi-GPU Programming</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Learn about Device Communication</div></li><li class="chapter-item expanded "><div>Version 1: Single GPU</div></li><li class="chapter-item expanded "><div>Version 2: Multi-GPU</div></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
