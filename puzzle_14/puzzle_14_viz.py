from manim import *

SIZE = 8
TPB = 8

class Puzzle14Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input array - use generic indices
        array = VGroup()
        for i in range(SIZE):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"a[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            array.add(cell)

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

        # Shared memory section
        shared_label = Text("Shared Memory (TPB=8)", font_size=14, color=WHITE)
        parallel_text = Text("Parallel prefix sum", font_size=14, color=YELLOW)
        shared_label_group = VGroup(shared_label, Text(" • ", font_size=14, color=WHITE), parallel_text)
        shared_label_group.arrange(RIGHT, buff=0.3)
        shared_label_group.next_to(threads, DOWN, buff=0.5)

        shared_mem = Rectangle(
            width=6.4,
            height=1,
            stroke_color=PURPLE_D,
            fill_color=PURPLE_E,
            fill_opacity=0.2
        ).next_to(shared_label_group, DOWN, buff=0.2)

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
            shared_label_group,
            shared_mem,
            shared_cells,
            block_label
        )
        block.move_to(ORIGIN)

        # Output array - use generic indices
        output_array = VGroup()
        for i in range(SIZE):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"out[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            output_array.add(cell)

        # Input array setup
        input_bg = Rectangle(
            width=array_scale * SIZE + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )
        array.arrange(RIGHT, buff=0.1)
        array.move_to(input_bg)
        array_group = VGroup(input_bg, array)
        array_label = Text("Input Array (size=8)", font_size=18)
        array_label.next_to(array_group, UP, buff=0.2)
        input_group = VGroup(array_label, array_group)
        input_group.to_edge(UP, buff=0.2)

        # Output array setup
        output_bg = Rectangle(
            width=array_scale * SIZE + 1.0,
            height=array_scale + 0.6,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )
        output_array.arrange(RIGHT, buff=0.1)
        output_array.move_to(output_bg)
        output_group = VGroup(output_bg, output_array)
        output_label = Text("Output Array (size=8)", font_size=18)
        output_label.next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)
        output_group.to_edge(DOWN, buff=0.2)

        # Initial animations - fix the order and reference correct groups
        self.play(
            Write(input_group)  # Use input_group instead of separate label and array
        )
        self.play(Create(block))
        self.play(
            Create(output_group)  # Use output_group directly
        )

        # Show data loading to shared memory
        load_arrows = VGroup()
        for i in range(SIZE):
            start = array[i].get_bottom()
            end = shared_cells[i].get_top()
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            load_arrows.add(arrow)

        self.play(FadeIn(load_arrows))
        self.wait(0.5)
        self.play(FadeOut(load_arrows))

        # Show reduction steps matching p12.mojo algorithm
        for step in range(3):  # log2(8) = 3 steps
            offset = 2 ** step
            arrows = VGroup()
            highlights = VGroup()

            for i in range(SIZE):
                if i >= offset:  # This matches the condition in p12.mojo
                    # Highlight active cells
                    h1 = Square(
                        side_length=cell_size,
                        stroke_color=YELLOW,
                        fill_opacity=0.2
                    ).move_to(shared_cells[i - offset])
                    h2 = Square(
                        side_length=cell_size,
                        stroke_color=YELLOW,
                        fill_opacity=0.2
                    ).move_to(shared_cells[i])
                    highlights.add(h1, h2)

                    # Curved arrows showing the addition pattern
                    curved_arrow = CurvedArrow(
                        start_point=shared_cells[i - offset].get_center(),
                        end_point=shared_cells[i].get_center(),
                        angle=-TAU/4,
                        color=GREEN_C,
                        stroke_width=2,
                        tip_length=0.2
                    )

                    # Operation text showing which values are being added
                    midpoint = (shared_cells[i].get_center() + shared_cells[i - offset].get_center()) / 2
                    op_text = Text("+", font_size=32, color=GREEN_C, weight=BOLD)
                    op_text.move_to(midpoint)
                    op_text.shift(DOWN * 0.5)

                    arrows.add(VGroup(curved_arrow, op_text))

            # Show step information matching the algorithm
            active_text = Text(f"Step {step + 1}: shared[i] += shared[i-{offset}] for i≥{offset}",
                             font_size=14, color=YELLOW)
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

        # Final output arrows remain straight
        output_arrows = VGroup()
        for i in range(SIZE):
            start = shared_cells[i].get_bottom()
            end = output_array[i].get_top()
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=GREEN_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            output_arrows.add(arrow)

        self.play(FadeIn(output_arrows))
        self.wait(0.5)
        self.play(FadeOut(output_arrows))

        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_14_viz"
    }):
        scene = Puzzle14Visualization()
        scene.render()
