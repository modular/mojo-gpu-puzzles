from manim import *

SIZE = 8
TPB = 8

class Puzzle12Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input arrays a and b
        input_arrays = VGroup()
        for name in ["a", "b"]:
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
                index_text = Text(f"{name}[{i}]", font_size=10, color=YELLOW)
                cell.add(index_text)
                array.add(cell)
            array.arrange(RIGHT, buff=0.1)
            array.move_to(array_bg)
            array_group = VGroup(array_bg, array)
            array_label = Text(f"Input Vector {name.upper()} (size=8)", font_size=18)
            array_label.next_to(array_group, UP, buff=0.2)
            input_arrays.add(VGroup(array_label, array_group))

        # Arrange arrays side by side with enough space between them
        input_arrays.arrange(RIGHT, buff=1.0)  # Increased buffer between arrays
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
        parallel_text = Text("Parallel reduction", font_size=14, color=YELLOW)
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

        # Output scalar
        output_bg = Rectangle(
            width=array_scale + 0.7,
            height=array_scale + 0.7,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )
        output_cell = Square(side_length=array_scale, stroke_width=1)
        output_text = Text("out[0]", font_size=10, color=YELLOW)
        output_cell.add(output_text)
        output_cell.move_to(output_bg)
        output_group = VGroup(output_bg, output_cell)
        output_label = Text("Output (size=1)", font_size=18).next_to(output_group, UP, buff=0.2)
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
            start = input_arrays[0][1][1][i].get_bottom()
            end = threads[i].get_top()
            arrow1 = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # b[i] to thread
            start = input_arrays[1][1][1][i].get_bottom()
            end = threads[i].get_top()
            arrow2 = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

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

            initial_arrows.add(arrow1, arrow2, arrow3)

        self.play(FadeIn(initial_arrows))
        self.wait(0.5)
        self.play(FadeOut(initial_arrows))

        # Show reduction steps
        stride = TPB // 2
        while stride > 0:
            arrows = VGroup()
            highlights = VGroup()

            for i in range(stride):
                # Highlight active cells
                h1 = Square(
                    side_length=cell_size,
                    stroke_color=YELLOW,
                    fill_opacity=0.2
                ).move_to(shared_cells[i])
                h2 = Square(
                    side_length=cell_size,
                    stroke_color=YELLOW,
                    fill_opacity=0.2
                ).move_to(shared_cells[i + stride])
                highlights.add(h1, h2)

                # Simple curved arrows for reduction
                curved_arrow = CurvedArrow(
                    start_point=shared_cells[i + stride].get_center(),
                    end_point=shared_cells[i].get_center(),
                    angle=-TAU/4,
                    color=GREEN_C,
                    stroke_width=2,
                    tip_length=0.2
                )

                # Add reduction operation text with fixed position
                midpoint = (shared_cells[i].get_center() + shared_cells[i + stride].get_center()) / 2
                op_text = Text("+", font_size=32, color=GREEN_C, weight=BOLD)
                op_text.move_to(midpoint)
                op_text.shift(DOWN * 0.5)  # Fixed offset down

                arrows.add(VGroup(curved_arrow, op_text))

            # Show active elements count
            active_text = Text(f"Active elements: {stride*2}", font_size=14, color=YELLOW)
            active_text.next_to(shared_cells, DOWN, buff=0.2)

            self.play(
                FadeIn(highlights),
                FadeIn(active_text)
            )
            self.play(FadeIn(arrows))
            self.wait(0.5)
            self.play(
                FadeOut(highlights),
                FadeOut(arrows),
                FadeOut(active_text)
            )

            stride //= 2

        # Final result arrow - simplified
        final_arrow = Arrow(
            shared_cells[0].get_bottom(),
            output_cell.get_top(),
            buff=0.2,
            color=GREEN_C,
            stroke_width=2
        )

        final_highlight = Square(
            side_length=cell_size,
            stroke_color=YELLOW,
            fill_opacity=0.2
        ).move_to(shared_cells[0])

        output_highlight = Square(
            side_length=array_scale,
            stroke_color=YELLOW,
            fill_opacity=0.2
        ).move_to(output_cell)

        self.play(FadeIn(final_highlight))
        self.play(
            FadeIn(final_arrow),
            FadeIn(output_highlight)
        )
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_12_viz"
    }):
        scene = Puzzle12Visualization()
        scene.render()
