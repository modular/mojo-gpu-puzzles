from manim import *

# Constants matching p08.mojo
SIZE = 8
TPB = 4

class Puzzle08Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input array (size=8)
        input_array = VGroup()
        input_bg = Rectangle(
            width=array_scale * SIZE + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )

        for i in range(SIZE):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"a[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_array.add(cell)
        input_array.arrange(RIGHT, buff=0.1)
        input_array.move_to(input_bg)
        input_group = VGroup(input_bg, input_array)
        input_label = Text("Input Array (size=8)", font_size=18).next_to(input_group, UP, buff=0.2)
        input_group = VGroup(input_label, input_group)
        input_group.to_edge(UP, buff=0.5)

        # GPU Blocks with Shared Memory
        blocks = VGroup()
        for block_idx in range(2):  # 2 blocks
            # Block background
            block_bg = Rectangle(
                width=4.2,
                height=3.6,
                stroke_color=GOLD_D,
                fill_color=DARK_GRAY,
                fill_opacity=0.1
            )

            # Threads first (at top)
            threads = VGroup()
            for i in range(TPB):
                thread_cell = RoundedRectangle(
                    width=0.8,
                    height=0.8,
                    corner_radius=0.1,
                    stroke_color=WHITE,
                    fill_color=DARK_GRAY,
                    fill_opacity=0.8
                )
                thread_text = Text(f"T{i}", font_size=12, color=YELLOW)
                thread_cell.add(thread_text)
                threads.add(thread_cell)
            threads.arrange(RIGHT, buff=0.25)
            threads.next_to(block_bg.get_top(), DOWN, buff=0.3)

            # Barrier sync indicator
            barrier_group = VGroup()
            barrier_line = DashedLine(
                start=threads.get_left() + LEFT * 0.3,
                end=threads.get_right() + RIGHT * 0.3,
                color=RED_D,
                dash_length=0.15
            ).next_to(threads, DOWN, buff=0.3)
            barrier_text = Text("barrier()", font_size=14, color=RED_D)
            barrier_text.next_to(barrier_line, DOWN, buff=0.15)
            barrier_group.add(barrier_line, barrier_text)

            # Shared memory section (below barrier)
            shared_label = Text(f"Shared Memory (TPB={TPB})", font_size=14, color=WHITE)
            shared_label.next_to(barrier_group, DOWN, buff=0.3)

            shared_mem = Rectangle(
                width=3.8,
                height=1.1,
                stroke_color=PURPLE_D,
                fill_color=PURPLE_E,
                fill_opacity=0.2
            ).next_to(shared_label, DOWN, buff=0.15)

            # Shared memory cells
            shared_cells = VGroup()
            for i in range(TPB):
                cell = Square(side_length=0.7, stroke_width=1, stroke_color=PURPLE_D)
                index_text = Text(f"shared[{i}]", font_size=10, color=YELLOW)
                cell.add(index_text)
                shared_cells.add(cell)
            shared_cells.arrange(RIGHT, buff=0.25)
            shared_cells.move_to(shared_mem)

            # Block label
            block_label = Text(f"Block {block_idx}", font_size=14, color=WHITE)
            if block_idx == 0:
                block_label.next_to(block_bg, UP, buff=0.2).shift(LEFT * 0.5)  # Subtle left shift
            else:
                block_label.next_to(block_bg, UP, buff=0.2).shift(RIGHT * 0.5)  # Subtle right shift

            block = VGroup(
                block_bg,
                threads,
                barrier_group,
                shared_label,
                shared_mem, shared_cells,
                block_label
            )
            blocks.add(block)

        blocks.arrange(RIGHT, buff=0.8)
        blocks.move_to(ORIGIN)

        # Output array - fix positioning
        output_bg = Rectangle(
            width=array_scale * SIZE + 1.0,
            height=array_scale + 0.6,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )

        output_array = VGroup()
        for i in range(SIZE):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"out[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            output_array.add(cell)
        output_array.arrange(RIGHT, buff=0.1)
        output_array.move_to(output_bg)
        output_group = VGroup(output_bg, output_array)
        output_label = Text("Output Array (size=8)", font_size=18).next_to(output_group, UP, buff=0.1)
        output_group = VGroup(output_label, output_group)
        output_group.shift(DOWN * 3.2)

        # Animations
        self.play(Write(input_label))
        self.play(Create(input_bg), Create(input_array), run_time=1.5)

        self.play(
            *[Create(block) for block in blocks],
            run_time=2
        )

        self.play(Write(output_label))
        self.play(Create(output_bg), Create(output_array), run_time=1.5)

        # Sample arrows showing data flow
        arrows = VGroup()

        # First block arrows (indices 0-3)
        for i in range(TPB):
            # Input to thread arrows
            start = input_array[i].get_bottom()
            end = blocks[0][1][i].get_top()  # threads
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            arrows.add(arrow)

            # Thread to shared memory arrows
            start = blocks[0][1][i].get_bottom()  # threads
            end = blocks[0][5][i].get_top()  # shared cells
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=PURPLE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            arrows.add(arrow)

            # Shared memory to output arrows
            start = blocks[0][5][i].get_bottom()  # shared cells
            end = output_array[i].get_top()
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=GREEN_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            arrows.add(arrow)

        # Second block arrows (indices 4-7)
        for i in range(TPB):
            # Input to thread arrows
            start = input_array[i + TPB].get_bottom()
            end = blocks[1][1][i].get_top()  # threads
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_A,  # Lighter blue
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            arrows.add(arrow)

            # Thread to shared memory arrows
            start = blocks[1][1][i].get_bottom()  # threads
            end = blocks[1][5][i].get_top()  # shared cells
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=PURPLE_A,  # Lighter purple
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            arrows.add(arrow)

            # Shared memory to output arrows
            start = blocks[1][5][i].get_bottom()  # shared cells
            end = output_array[i + TPB].get_top()
            arrow = Arrow(
                start, end,
                buff=0.2,
                color=GREEN_A,  # Lighter green
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)
            arrows.add(arrow)

        self.play(FadeIn(arrows), run_time=0.8)
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_08_viz"
    }):
        scene = Puzzle08Visualization()
        scene.render()
