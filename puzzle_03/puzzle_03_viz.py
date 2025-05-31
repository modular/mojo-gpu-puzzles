from manim import *

class Puzzle03Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input array
        input_array = VGroup()
        input_bg = Rectangle(
            width=array_scale * 4 + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )

        for i in range(4):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"a[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_array.add(cell)
        input_array.arrange(RIGHT, buff=0)
        input_array.move_to(input_bg)
        input_group = VGroup(input_bg, input_array)
        input_label = Text("Input Array (size=4)", font_size=18).next_to(input_group, UP, buff=0.2)
        input_group = VGroup(input_label, input_group)
        input_group.shift(UP * 2.5)

        # GPU Thread Block
        block_bg = Rectangle(
            width=13,
            height=2,
            stroke_color=GOLD_D,
            fill_color=DARK_GRAY,
            fill_opacity=0.1
        )
        block_label = Text("GPU Parallel Threads in a Block", font_size=18).next_to(block_bg, UP, buff=0.2)

        threads = VGroup()
        for i in range(8):  # 8 threads
            thread_cell = RoundedRectangle(
                width=1.4,
                height=1.0,
                corner_radius=0.1,
                stroke_color=WHITE,
                fill_color=DARK_GRAY,
                fill_opacity=0.8
            )
            thread_text = Text(f"thread_idx.x={i}", font_size=14, color=YELLOW)
            valid_text = Text("if i < size", font_size=10, color=GREEN_A if i < 4 else RED)
            thread_info = VGroup(thread_text, valid_text).arrange(DOWN, buff=0.05)
            thread_cell.add(thread_info)
            threads.add(thread_cell)

        threads.arrange(RIGHT, buff=0.15)
        threads.move_to(block_bg)
        threads.shift(UP * 0.1)

        block_group = VGroup(block_bg, block_label, threads)
        block_group.move_to(ORIGIN)

        # Output array
        output_array = VGroup()
        output_bg = Rectangle(
            width=array_scale * 4 + 1.5,
            height=array_scale + 1,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        ).shift(DOWN * 0.5)

        for i in range(4):
            cell = Square(side_length=array_scale * 1.2, stroke_width=1)
            index_text = Text(f"out[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            output_array.add(cell)
        output_array.arrange(RIGHT, buff=0)
        output_array.move_to(output_bg)
        output_group = VGroup(output_bg, output_array)
        output_label = Text("Output Array (size=4)", font_size=18).next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)
        output_group.shift(DOWN * 2.5)

        # Animations
        self.play(Write(input_label))
        self.play(Create(input_bg), Create(input_array), run_time=1.5)

        self.play(Write(block_label))
        self.play(Create(block_bg), Create(threads), run_time=1.5)

        self.play(Write(output_label))
        self.play(Create(output_bg), Create(output_array), run_time=1.5)

        # Create arrows flowing top to bottom
        arrows = VGroup()
        for i in range(8):
            if i < 4:  # Only create arrows for valid threads
                # Arrow from input to thread
                start = input_array[i].get_bottom()
                end = threads[i].get_top()
                arrow1 = Arrow(
                    start, end,
                    buff=0.2,
                    color=BLUE_C,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.2
                ).set_opacity(0.6)

                # Arrow from thread to output
                start = threads[i].get_bottom()
                end = output_array[i].get_top()
                arrow2 = Arrow(
                    start, end,
                    buff=0.2,
                    color=GREEN_C,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.2
                ).set_opacity(0.6)

                arrows.add(arrow1, arrow2)
            else:
                # Add X marks for invalid threads
                warning = Text("×", font_size=36, color=RED).move_to(threads[i])
                arrows.add(warning)

        self.play(FadeIn(arrows), run_time=0.3)
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_03_viz"
    }):
        scene = Puzzle03Visualization()
        scene.render()
