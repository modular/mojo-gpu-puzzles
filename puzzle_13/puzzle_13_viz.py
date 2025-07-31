from manim import *

SIZE = 6  # Match the example size
CONV = 3  # Convolution kernel size
TPB = 8

class Puzzle13Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input arrays a and b (kernel)
        input_arrays = VGroup()

        # Input array a
        array_bg = Rectangle(
            width=array_scale * SIZE + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )
        array = VGroup()
        for i in range(SIZE):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"a[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            array.add(cell)
        array.arrange(RIGHT, buff=0.1)
        array.move_to(array_bg)
        array_group = VGroup(array_bg, array)
        array_label = Text(f"Input Array (size={SIZE})", font_size=18)
        array_label.next_to(array_group, UP, buff=0.2)
        input_arrays.add(VGroup(array_label, array_group))

        # Kernel array b
        kernel_bg = Rectangle(
            width=array_scale * CONV + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )
        kernel = VGroup()
        for i in range(CONV):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"b[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            kernel.add(cell)
        kernel.arrange(RIGHT, buff=0.1)
        kernel.move_to(kernel_bg)
        kernel_group = VGroup(kernel_bg, kernel)
        kernel_label = Text(f"Convolution Kernel (size={CONV})", font_size=18)
        kernel_label.next_to(kernel_group, UP, buff=0.2)
        input_arrays.add(VGroup(kernel_label, kernel_group))

        # Arrange arrays side by side
        input_arrays.arrange(RIGHT, buff=1.0)
        input_arrays.to_edge(UP, buff=0.2)

        # GPU Block
        block_bg = Rectangle(
            width=7,
            height=3.7,
            stroke_color=GOLD_D,
            fill_color=DARK_GRAY,
            fill_opacity=0.1
        )

        # Threads
        threads = VGroup()
        for i in range(TPB):
            thread_cell = RoundedRectangle(
                width=0.6,
                height=0.8,
                corner_radius=0.1,
                stroke_color=WHITE,
                fill_color=DARK_GRAY,
                fill_opacity=0.8
            )
            thread_text = Text(f"T{i}", font_size=12, color=YELLOW)
            thread_cell.add(thread_text)
            threads.add(thread_cell)
        threads.arrange(RIGHT, buff=0.2)
        threads.next_to(block_bg.get_top(), DOWN, buff=0.3)

        # Barrier sync
        barrier_group = VGroup()
        barrier_line = DashedLine(
            start=threads.get_left() + LEFT * 0.8,
            end=threads.get_right() + RIGHT * 0.8,
            color=RED_D,
            dash_length=0.15
        ).next_to(threads, DOWN, buff=0.3)
        barrier_text = Text("barrier()", font_size=14, color=RED_D)
        barrier_text.next_to(barrier_line, DOWN, buff=0.15)
        barrier_group.add(barrier_line, barrier_text)

        # Shared memory
        shared_label = Text("Shared Memory (TPB=8)", font_size=14, color=WHITE)
        parallel_text = Text("Sliding window convolution", font_size=14, color=YELLOW)
        shared_label_group = VGroup(shared_label, Text(" • ", font_size=14, color=WHITE), parallel_text)
        shared_label_group.arrange(RIGHT, buff=0.3)
        shared_label_group.next_to(barrier_group, DOWN, buff=0.3)

        shared_mem = Rectangle(
            width=6.4,
            height=1,
            stroke_color=PURPLE_D,
            fill_color=PURPLE_E,
            fill_opacity=0.2
        ).next_to(shared_label_group, DOWN, buff=0.15)

        # Shared memory cells
        shared_cells = VGroup()
        cell_size = 0.7
        for i in range(TPB):
            cell = Square(side_length=cell_size, stroke_width=1, stroke_color=PURPLE_D)
            index_text = Text(f"shared[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            shared_cells.add(cell)
        shared_cells.arrange(RIGHT, buff=0)
        shared_cells.move_to(shared_mem)

        block_label = Text("Block 0", font_size=14, color=WHITE)
        block_label.next_to(block_bg, UP, buff=0.2)

        block = VGroup(
            block_bg,
            threads,
            barrier_group,
            shared_label_group,
            shared_mem, shared_cells,
            block_label
        )
        block.move_to(ORIGIN)

        # Output array - use full SIZE
        output_bg = Rectangle(
            width=array_scale * SIZE + 1.0,  # Use full SIZE
            height=array_scale + 0.6,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )
        output_array = VGroup()
        for i in range(SIZE):  # Use full SIZE
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"out[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            output_array.add(cell)
        output_array.arrange(RIGHT, buff=0.1)
        output_array.move_to(output_bg)
        output_group = VGroup(output_bg, output_array)
        output_label = Text(f"Output Array (size={SIZE})", font_size=18).next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)
        output_group.to_edge(DOWN, buff=0.2)

        # Initial animations
        self.play(
            Write(input_arrays[0][0]),
            Write(input_arrays[1][0])
        )
        self.play(
            Create(input_arrays[0][1]),
            Create(input_arrays[1][1])
        )
        self.play(Create(block))
        self.play(Create(output_group))

        # Show parallel loading
        initial_arrows = VGroup()
        for i in range(SIZE):
            # a[i] to thread
            start = array[i].get_bottom()
            end = threads[i].get_top()
            arrow1 = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # kernel to thread if in range
            if i < CONV:
                start = kernel[i].get_bottom()
                end = threads[i].get_top()
                arrow2 = Arrow(
                    start, end,
                    buff=0.2,
                    color=BLUE_C,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.2
                ).set_opacity(0.6)
                initial_arrows.add(arrow2)

            # Thread to shared memory
            start = threads[i].get_bottom()
            end = shared_cells[i].get_top()
            arrow3 = Arrow(
                start, end,
                buff=0.2,
                color=PURPLE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            initial_arrows.add(arrow1, arrow3)

        self.play(FadeIn(initial_arrows))
        self.wait(0.5)
        self.play(FadeOut(initial_arrows))

        # Show convolution window sliding
        window_highlight = Rectangle(
            width=cell_size * CONV,
            height=cell_size + 0.2,
            stroke_color=YELLOW,
            fill_opacity=0.2
        )
        window_label = Text("Window size=3", font_size=14, color=YELLOW)
        window_label.next_to(window_highlight, DOWN, buff=0.1)

        # Slide through all positions including partial windows
        for pos in range(SIZE):
            # Calculate available elements for this window
            remaining = SIZE - pos  # How many elements remain from this position
            window_size = min(CONV, remaining)  # Take min of CONV or remaining elements

            if window_size == 0:  # Skip if no elements available
                break

            window_cells = VGroup(*[shared_cells[i] for i in range(pos, pos + window_size)])

            # Center window highlight on available cells
            window_highlight.move_to(window_cells.get_center())
            if window_size < CONV:
                # Adjust width for partial windows
                window_highlight.stretch_to_fit_width(cell_size * window_size)

            # Show multiplication and sum
            arrows = VGroup()

            # Show multiplications for available elements
            for i in range(window_size):
                mult_text = Text("×", font_size=32, color=GREEN_C, weight=BOLD, stroke_width=2)
                mult_text.next_to(shared_cells[pos + i], UP, buff=0.3)
                arrows.add(mult_text)

            # Plus symbol
            plus_text = Text("+", font_size=36, color=GREEN_C, weight=BOLD, stroke_width=2)
            plus_text.next_to(window_highlight, DOWN, buff=0.3)
            arrows.add(plus_text)

            # Output arrow
            output_arrow = Arrow(
                plus_text.get_bottom(),
                output_array[pos].get_top(),
                buff=0.2,
                color=GREEN_C,
                stroke_width=4
            )
            arrows.add(output_arrow)

            if pos == 0:
                self.play(
                    Create(window_highlight),
                    Write(window_label)
                )
            else:
                self.play(
                    Transform(window_highlight, window_highlight.copy()),
                    window_label.animate.next_to(window_highlight, DOWN, buff=0.1)
                )

            self.play(FadeIn(arrows))
            self.wait(0.3)
            self.play(FadeOut(arrows))

        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_13_viz"
    }):
        scene = Puzzle13Visualization()
        scene.render()
