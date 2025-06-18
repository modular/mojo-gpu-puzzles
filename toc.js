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
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="introduction.html">🔥 Introduction</a></li><li class="chapter-item expanded "><a href="howto.html">🧭 Puzzles Usage Guide</a></li><li class="chapter-item expanded affix "><li class="part-title">Part I: GPU Fundamentals</li><li class="chapter-item expanded "><a href="puzzle_01/puzzle_01.html">Puzzle 1: Map</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_01/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_01/layout_tensor_preview.html">💡 Preview: Modern Approach with LayoutTensor</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_02/puzzle_02.html">Puzzle 2: Zip</a></li><li class="chapter-item expanded "><a href="puzzle_03/puzzle_03.html">Puzzle 3: Guards</a></li><li class="chapter-item expanded "><a href="puzzle_04/puzzle_04.html">Puzzle 4: 2D Map</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_04/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_04/introduction_layout_tensor.html">📚 Learn about LayoutTensor</a></li><li class="chapter-item expanded "><a href="puzzle_04/layout_tensor.html">🚀 Modern 2D Operations</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_05/puzzle_05.html">Puzzle 5: Broadcast</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_05/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_05/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_06/puzzle_06.html">Puzzle 6: Blocks</a></li><li class="chapter-item expanded "><a href="puzzle_07/puzzle_07.html">Puzzle 7: 2D Blocks</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_07/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_07/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_08/puzzle_08.html">Puzzle 8: Shared Memory</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_08/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_08/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part II: 🧮 GPU Algorithms</li><li class="chapter-item expanded "><a href="puzzle_09/puzzle_09.html">Puzzle 9: Pooling</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_09/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_09/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_10/puzzle_10.html">Puzzle 10: Dot Product</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_10/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_10/layout_tensor.html">📐 LayoutTensor Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_11/puzzle_11.html">Puzzle 11: 1D Convolution</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_11/simple.html">🔰 Simple Version</a></li><li class="chapter-item expanded "><a href="puzzle_11/block_boundary.html">⭐ Block Boundary Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_12/puzzle_12.html">Puzzle 12: Prefix Sum</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_12/simple.html">🔰 Simple Version</a></li><li class="chapter-item expanded "><a href="puzzle_12/complete.html">⭐ Complete Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_13/puzzle_13.html">Puzzle 13: Axis Sum</a></li><li class="chapter-item expanded "><a href="puzzle_14/puzzle_14.html">Puzzle 14: Matrix Multiplication (MatMul)</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_14/naïve.html">🔰 Naïve Version with Global Memory</a></li><li class="chapter-item expanded "><a href="puzzle_14/roofline.html">📚 Learn about Roofline Model</a></li><li class="chapter-item expanded "><a href="puzzle_14/shared_memory.html">🤝 Shared Memory Version</a></li><li class="chapter-item expanded "><a href="puzzle_14/tiled.html">📐 Tiled Version</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part III: 🐍 Interfacing with Python via MAX Graph Custom Ops</li><li class="chapter-item expanded "><a href="puzzle_15/puzzle_15.html">Puzzle 15: 1D Convolution Op</a></li><li class="chapter-item expanded "><a href="puzzle_16/puzzle_16.html">Puzzle 16: Softmax Op</a></li><li class="chapter-item expanded "><a href="puzzle_17/puzzle_17.html">Puzzle 17: Attention Op</a></li><li class="chapter-item expanded "><a href="bonuses/part3.html">🎯 Bonus Challenges</a></li><li class="chapter-item expanded affix "><li class="part-title">Part IV: 🔥 PyTorch Custom Ops Integration</li><li class="chapter-item expanded "><a href="puzzle_18/puzzle_18.html">Puzzle 18: 1D Convolution Op</a></li><li class="chapter-item expanded "><a href="puzzle_19/puzzle_19.html">Puzzle 19: Embedding Op</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_19/simple_embedding_kernel.html">🔰 Coaleasced vs non-Coaleasced Kernel</a></li><li class="chapter-item expanded "><a href="puzzle_19/performance.html">📊 Performance Comparison</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_20/puzzle_20.html">Puzzle 20: Kernel Fusion and Custom Backward Pass</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_20/forward_pass.html">⚛️ Fused vs Unfused Kernels</a></li><li class="chapter-item expanded "><a href="puzzle_20/backward_pass.html">⛓️ Autograd Integration &amp; Backward Pass</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part V: 🌊 Mojo Functional Patterns and Benchmarking</li><li class="chapter-item expanded "><a href="puzzle_21/puzzle_21.html">Puzzle 21: GPU Functional Programming Patterns</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_21/elementwise.html">elementwise - Basic GPU Functional Operations</a></li><li class="chapter-item expanded "><a href="puzzle_21/tile.html">tile - Memory-Efficient Tiled Processing</a></li><li class="chapter-item expanded "><a href="puzzle_21/vectorize.html">Vectorization - SIMD Control</a></li><li class="chapter-item expanded "><a href="puzzle_21/gpu-thread-vs-simd.html">🧠 GPU Threading vs SIMD Concepts</a></li><li class="chapter-item expanded "><a href="puzzle_21/benchmarking.html">📊 Benchmarking in Mojo</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part VI: ⚡ Warp-Level Programming</li><li class="chapter-item expanded "><a href="puzzle_22/puzzle_22.html">Puzzle 22: Warp Fundamentals</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_22/warp_simt.html">🧠 Warp lanes &amp; SIMT execution</a></li><li class="chapter-item expanded "><a href="puzzle_22/warp_sum.html">🔰 warp.sum() Essentials</a></li><li class="chapter-item expanded "><a href="puzzle_22/warp_extra.html">📊 When to Use Warp Programming</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_23/puzzle_23.html">Puzzle 23: Warp Communication</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_23/warp_shuffle_down.html">⬇️ warp.shuffle_down()</a></li><li class="chapter-item expanded "><a href="puzzle_23/warp_broadcast.html">📢 warp.broadcast()</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_24/puzzle_24.html">Puzzle 24: Advanced Warp Patterns</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_24/warp_shuffle_xor.html">🦋 warp.shuffle_xor() Butterfly Networks</a></li><li class="chapter-item expanded "><a href="puzzle_24/warp_prefix_sum.html">🔢 warp.prefix_sum() Scan Operations</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part VII: Advanced Memory Operations</li><li class="chapter-item expanded "><div>Puzzle 25: Memory Coalescing</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Understanding Coalesced Access</div></li><li class="chapter-item expanded "><div>Optimized Access Patterns</div></li><li class="chapter-item expanded "><div>🔧 Troubleshooting Memory Issues</div></li></ol></li><li class="chapter-item expanded "><div>Puzzle 26: Async Memory Operations</div></li><li class="chapter-item expanded "><div>Puzzle 27: Memory Fences &amp; Atomics</div></li><li class="chapter-item expanded "><div>Puzzle 28: Prefetching &amp; Caching</div></li><li class="chapter-item expanded affix "><li class="part-title">Part VIII: 📊 Performance Analysis &amp; Optimization</li><li class="chapter-item expanded "><div>Puzzle 29: GPU Profiling Basics</div></li><li class="chapter-item expanded "><div>Puzzle 30: Occupancy Optimization</div></li><li class="chapter-item expanded "><div>Puzzle 31: Bank Conflicts</div></li><li><ol class="section"><li class="chapter-item expanded "><div>📚 Understanding Shared Memory Banks</div></li><li class="chapter-item expanded "><div>Conflict-Free Patterns</div></li></ol></li><li class="chapter-item expanded "><li class="part-title">Part IX: 🚀 Advanced GPU Features</li><li class="chapter-item expanded "><div>Puzzle 32: Tensor Core Operations</div></li><li class="chapter-item expanded "><div>Puzzle 33: Random Number Generation</div></li><li class="chapter-item expanded "><div>Puzzle 34: Advanced Synchronization</div></li><li class="chapter-item expanded affix "><li class="part-title">Part X: 🌐 Multi-GPU &amp; Advanced Applications</li><li class="chapter-item expanded "><div>Puzzle 35: Multi-Stream Programming</div></li><li class="chapter-item expanded "><div>Puzzle 36: Multi-GPU Basics</div></li><li class="chapter-item expanded "><div>Puzzle 37: End-to-End Optimization Case Study</div></li><li class="chapter-item expanded "><div>🎯 Advanced Bonus Challenges</div></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
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
