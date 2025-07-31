from manim import *

SIZE = 8
TPB = 8
WINDOW_SIZE = 3

class Puzzle11Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input array
        input_array = VGroup()
        input_bg = Rectangle(
            width=array_scale * SIZE + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        ).shift(UP * 0.5)

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
        input_group.to_edge(UP, buff=0.2)

        # GPU Block with Shared Memory
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
            start=threads.get_left() + LEFT * 0.6,
            end=threads.get_right() + RIGHT * 0.6,
            color=RED_D,
            dash_length=0.15
        ).next_to(threads, DOWN, buff=0.3)
        barrier_text = Text("barrier()", font_size=14, color=RED_D)
        barrier_text.next_to(barrier_line, DOWN, buff=0.15)
        barrier_group.add(barrier_line, barrier_text)

        # Shared memory section - make box bigger
        shared_label = Text("Shared Memory (TPB=8)", font_size=14, color=WHITE)
        parallel_text = Text("All windows process in parallel", font_size=14, color=YELLOW)

        # Arrange shared_label and parallel_text side by side
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

        # Shared memory cells - make contiguous
        shared_cells = VGroup()
        cell_size = 0.7  # Keep consistent cell size
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

        # Output array
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
        output_label = Text("Output Array (size=8)", font_size=18).next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)
        output_group.to_edge(DOWN, buff=0.2)

        # Animations
        self.play(Write(input_label))
        self.play(Create(input_bg), Create(input_array))
        self.play(Create(block))
        self.play(Write(output_label))
        self.play(Create(output_bg), Create(output_array))

        # First show parallel data loading
        initial_arrows = VGroup()
        for i in range(SIZE):
            # Input to thread
            start = input_array[i].get_bottom()
            end = threads[i].get_top()
            arrow1 = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # Thread to shared memory
            start = threads[i].get_bottom()
            end = shared_cells[i].get_top()
            arrow2 = Arrow(
                start, end,
                buff=0.2,
                color=PURPLE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            initial_arrows.add(arrow1, arrow2)

        self.play(FadeIn(initial_arrows))
        self.wait(0.5)
        self.play(FadeOut(initial_arrows))

        # Window animation sequence over shared memory
        window_highlight = Rectangle(
            width=cell_size * WINDOW_SIZE,  # Exact width for 3 cells
            height=cell_size + 0.2,  # Slightly taller than cells
            stroke_color=YELLOW,
            fill_opacity=0.2
        )
        window_label = Text("Window size=3", font_size=14, color=YELLOW)
        window_label.next_to(window_highlight, UP, buff=0.1)

        # Output value highlights
        output_highlights = VGroup()

        for pos in range(SIZE):
            if pos == 0:
                # First element: just shared[0]
                window_highlight.move_to(shared_cells[0])
                window_highlight.shift(LEFT * cell_size)  # Space to left
                window_cells = VGroup(shared_cells[0])
            elif pos == 1:
                # Second element: shared[0], shared[1]
                window_cells = VGroup(shared_cells[0:2])
                window_highlight.move_to(window_cells.get_center())
                window_highlight.shift(LEFT * cell_size * 0.5)  # Space to left
            else:
                # Rest: sliding window of 3 over shared memory
                window_cells = VGroup(shared_cells[pos-2:pos+1])
                window_highlight.move_to(window_cells.get_center())

            output_highlight = Square(
                side_length=array_scale,
                stroke_color=YELLOW,
                fill_opacity=0.2
            ).move_to(output_array[pos])

            window_label.next_to(window_highlight, UP, buff=0.1)

            # Clear previous arrows
            self.remove(*[obj for obj in self.mobjects if isinstance(obj, Arrow)])

            arrows = VGroup()

            # Shared memory to output arrows - different for each case
            if pos == 0:
                arrow = Arrow(
                    shared_cells[0].get_bottom(),
                    output_array[0].get_top(),
                    buff=0.2,
                    color=GREEN_C,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.2
                ).set_opacity(0.6)
                arrows.add(arrow)
            elif pos == 1:
                for i in range(2):
                    arrow = Arrow(
                        shared_cells[i].get_bottom(),
                        output_array[1].get_top(),
                        buff=0.2,
                        color=GREEN_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)
                    arrows.add(arrow)
            else:
                for i in range(pos-2, pos+1):
                    arrow = Arrow(
                        shared_cells[i].get_bottom(),
                        output_array[pos].get_top(),
                        buff=0.2,
                        color=GREEN_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)
                    arrows.add(arrow)

            # Animation
            if pos == 0:
                self.play(
                    Create(window_highlight),
                    Write(window_label),
                    FadeIn(arrows),
                    Create(output_highlight)
                )
            else:
                # Store previous arrows to fade them out
                prev_arrows = self.mobjects[:]  # Get all current mobjects
                prev_arrows = [m for m in prev_arrows if isinstance(m, Arrow)]  # Filter arrows

                self.play(
                    Transform(window_highlight, window_highlight.copy()),
                    window_label.animate.next_to(window_highlight, UP, buff=0.1),
                    *[FadeOut(arrow) for arrow in prev_arrows],  # Fade out all previous arrows
                    FadeIn(arrows),
                    Create(output_highlight),
                    run_time=0.8
                )

            output_highlights.add(output_highlight)
            self.wait(0.5)

        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_11_viz"
    }):
        scene = Puzzle11Visualization()
        scene.render()
