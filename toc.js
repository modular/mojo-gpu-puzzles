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
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="introduction.html">🔥 Introduction</a></li><li class="chapter-item expanded "><a href="howto.html">🧭 Puzzles Usage Guide</a></li><li class="chapter-item expanded "><a href="reward.html">🏆 Claim Your Rewards</a></li><li class="chapter-item expanded "><a href="puzzle_01/puzzle_01.html">Puzzle 1: Map</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_01/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_01/tile_tensor_preview.html">💡 Preview: Modern Approach with TileTensor</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_02/puzzle_02.html">Puzzle 2: Zip</a></li><li class="chapter-item expanded "><a href="puzzle_03/puzzle_03.html">Puzzle 3: Guards</a></li><li class="chapter-item expanded "><a href="puzzle_04/puzzle_04.html">Puzzle 4: 2D Map</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_04/raw.html">🔰 Raw Memory Approach</a></li><li class="chapter-item expanded "><a href="puzzle_04/introduction_tile_tensor.html">📚 Learn about TileTensor</a></li><li class="chapter-item expanded "><a href="puzzle_04/tile_tensor.html">🚀 Modern 2D Operations</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_05/puzzle_05.html">Puzzle 5: Broadcast</a></li><li class="chapter-item expanded "><a href="puzzle_06/puzzle_06.html">Puzzle 6: Blocks</a></li><li class="chapter-item expanded "><a href="puzzle_07/puzzle_07.html">Puzzle 7: 2D Blocks</a></li><li class="chapter-item expanded "><a href="puzzle_08/puzzle_08.html">Puzzle 8: Shared Memory</a></li><li class="chapter-item expanded "><a href="puzzle_09/puzzle_09.html">Puzzle 9: GPU Debugging Workflow</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_09/essentials.html">📚 Mojo GPU Debugging Essentials</a></li><li class="chapter-item expanded "><a href="puzzle_09/first_case.html">🧐 Detective Work: First Case</a></li><li class="chapter-item expanded "><a href="puzzle_09/second_case.html">🔍 Detective Work: Second Case</a></li><li class="chapter-item expanded "><a href="puzzle_09/third_case.html">🕵 Detective Work: Third Case</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_10/puzzle_10.html">Puzzle 10: Memory Error Detection &amp; Race Conditions with Sanitizers</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_10/memcheck.html">👮🏼‍♂️ Detect Memory Violation</a></li><li class="chapter-item expanded "><a href="puzzle_10/racecheck.html">🏁 Debug Race Condition</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_11/puzzle_11.html">Puzzle 11: Pooling</a></li><li class="chapter-item expanded "><a href="puzzle_12/puzzle_12.html">Puzzle 12: Dot Product</a></li><li class="chapter-item expanded "><a href="puzzle_13/puzzle_13.html">Puzzle 13: 1D Convolution</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_13/simple.html">🔰 Simple Version</a></li><li class="chapter-item expanded "><a href="puzzle_13/block_boundary.html">⭐ Block Boundary Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_14/puzzle_14.html">Puzzle 14: Prefix Sum</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_14/simple.html">🔰 Simple Version</a></li><li class="chapter-item expanded "><a href="puzzle_14/complete.html">⭐ Complete Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_15/puzzle_15.html">Puzzle 15: Axis Sum</a></li><li class="chapter-item expanded "><a href="puzzle_16/puzzle_16.html">Puzzle 16: Matrix Multiplication (MatMul)</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_16/naïve.html">🔰 Naïve Version with Global Memory</a></li><li class="chapter-item expanded "><a href="puzzle_16/roofline.html">📚 Learn about Roofline Model</a></li><li class="chapter-item expanded "><a href="puzzle_16/shared_memory.html">🤝 Shared Memory Version</a></li><li class="chapter-item expanded "><a href="puzzle_16/tiled.html">📐 Tiled Version</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_17/puzzle_17.html">Puzzle 17: 1D Convolution Op</a></li><li class="chapter-item expanded "><a href="puzzle_18/puzzle_18.html">Puzzle 18: Softmax Op</a></li><li class="chapter-item expanded "><a href="puzzle_19/puzzle_19.html">Puzzle 19: Attention Op</a></li><li class="chapter-item expanded "><a href="bonuses/part4.html">🎯 Bonus Challenges</a></li><li class="chapter-item expanded "><a href="puzzle_20/puzzle_20.html">Puzzle 20: 1D Convolution Op</a></li><li class="chapter-item expanded "><a href="puzzle_21/puzzle_21.html">Puzzle 21: Embedding Op</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_21/simple_embedding_kernel.html">🔰 Coalesced vs Non-Coalesced Kernel</a></li><li class="chapter-item expanded "><a href="puzzle_21/performance.html">📊 Performance Comparison</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_22/puzzle_22.html">Puzzle 22: Kernel Fusion and Custom Backward Pass</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_22/forward_pass.html">⚛️ Fused vs Unfused Kernels</a></li><li class="chapter-item expanded "><a href="puzzle_22/backward_pass.html">⛓️ Autograd Integration &amp; Backward Pass</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_23/puzzle_23.html">Puzzle 23: GPU Functional Programming Patterns</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_23/elementwise.html">elementwise - Basic GPU Functional Operations</a></li><li class="chapter-item expanded "><a href="puzzle_23/tile.html">tile - Memory-Efficient Tiled Processing</a></li><li class="chapter-item expanded "><a href="puzzle_23/vectorize.html">vectorize - SIMD Control</a></li><li class="chapter-item expanded "><a href="puzzle_23/gpu-thread-vs-simd.html">🧠 GPU Threading vs SIMD Concepts</a></li><li class="chapter-item expanded "><a href="puzzle_23/benchmarking.html">📊 Benchmarking in Mojo</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_24/puzzle_24.html">Puzzle 24: Warp Fundamentals</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_24/warp_simt.html">🧠 Warp lanes &amp; SIMT execution</a></li><li class="chapter-item expanded "><a href="puzzle_24/warp_sum.html">🔰 warp.sum() Essentials</a></li><li class="chapter-item expanded "><a href="puzzle_24/warp_extra.html">🤔 When to Use Warp Programming</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_25/puzzle_25.html">Puzzle 25: Warp Communication</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_25/warp_shuffle_down.html">⬇️ warp.shuffle_down()</a></li><li class="chapter-item expanded "><a href="puzzle_25/warp_broadcast.html">📢 warp.broadcast()</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_26/puzzle_26.html">Puzzle 26: Advanced Warp Patterns</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_26/warp_shuffle_xor.html">🦋 warp.shuffle_xor() Butterfly Networks</a></li><li class="chapter-item expanded "><a href="puzzle_26/warp_prefix_sum.html">🔢 warp.prefix_sum() Scan Operations</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_27/puzzle_27.html">Puzzle 27: Block-Wide Patterns</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_27/block_sum.html">🔰 block.sum() Essentials</a></li><li class="chapter-item expanded "><a href="puzzle_27/block_prefix_sum.html">📈 block.prefix_sum() Parallel Histogram Binning</a></li><li class="chapter-item expanded "><a href="puzzle_27/block_broadcast.html">📡 block.broadcast() Vector Normalization</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_28/puzzle_28.html">Puzzle 28: Async Memory Operations &amp; Copy Overlap</a></li><li class="chapter-item expanded "><a href="puzzle_29/puzzle_29.html">Puzzle 29: GPU Synchronization Primitives</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_29/barrier.html">📶 Multi-Stage Pipeline Coordination</a></li><li class="chapter-item expanded "><a href="puzzle_29/memory_barrier.html">Double-Buffered Stencil Computation</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_30/puzzle_30.html">Puzzle 30: GPU Profiling</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_30/nvidia_profiling_basics.html">📚 NVIDIA Profiling Basics</a></li><li class="chapter-item expanded "><a href="puzzle_30/profile_kernels.html">🕵 The Cache Hit Paradox</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_31/puzzle_31.html">Puzzle 31: Occupancy Optimization</a></li><li class="chapter-item expanded "><a href="puzzle_32/puzzle_32.html">Puzzle 32: Bank Conflicts</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_32/shared_memory_bank.html">📚 Understanding Shared Memory Banks</a></li><li class="chapter-item expanded "><a href="puzzle_32/conflict_free_patterns.html">Conflict-Free Patterns</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_33/puzzle_33.html">Puzzle 33: Tensor Core Operations</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="bonuses/part5.html">🎯 Performance Bonus Challenge</a></li></ol></li><li class="chapter-item expanded "><a href="puzzle_34/puzzle_34.html">Puzzle 34: GPU Cluster Programming (SM90+)</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="puzzle_34/cluster_coordination_basics.html">🔰 Multi-Block Coordination Basics</a></li><li class="chapter-item expanded "><a href="puzzle_34/cluster_collective_ops.html">☸️ Cluster-Wide Collective Operations</a></li><li class="chapter-item expanded "><a href="puzzle_34/advanced_cluster_patterns.html">🧠 Advanced Cluster Algorithms</a></li></ol></li></ol>';
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
