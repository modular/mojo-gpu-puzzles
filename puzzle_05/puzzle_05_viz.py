from manim import *

class Puzzle05Visualization(Scene):
    def construct(self):
        array_scale = 0.6
        thread_scale = 0.4

        # Input vector A (1x2) - positioned on the left
        input_vector_a = VGroup()
        input_bg_a = Rectangle(
            width=array_scale * 2 + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        ).shift(UP * 2)

        for i in range(2):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"a[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_vector_a.add(cell)
        input_vector_a.arrange(RIGHT, buff=0)
        input_vector_a.move_to(input_bg_a)
        input_group_a = VGroup(input_bg_a, input_vector_a)
        input_label_a = Text("Input Vector A (size=2)", font_size=18).next_to(input_group_a, UP, buff=0.2)
        input_group_a = VGroup(input_label_a, input_group_a)
        input_group_a.to_edge(LEFT, buff=1.0)

        # Input vector B (2x1) - positioned on the left, below A
        input_vector_b = VGroup()
        input_bg_b = Rectangle(
            width=array_scale + 0.6,
            height=array_scale * 2 + 1.0,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        ).shift(UP * 2)

        for i in range(2):
            cell = Square(side_length=array_scale + 0.1, stroke_width=1)
            index_text = Text(f"b[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_vector_b.add(cell)
        input_vector_b.arrange(DOWN, buff=0)
        input_vector_b.move_to(input_bg_b)
        input_group_b = VGroup(input_bg_b, input_vector_b)
        input_label_b = Text("Input Vector B (size=2)", font_size=18).next_to(input_group_b, UP, buff=0.2)
        input_group_b = VGroup(input_label_b, input_group_b)
        input_group_b.next_to(input_group_a, DOWN, buff=1.5)

        # GPU Thread Block - make it 3x3
        block_bg = Rectangle(
            width=6,
            height=6,
            stroke_color=GOLD_D,
            fill_color=DARK_GRAY,
            fill_opacity=0.1
        )
        block_label = Text("GPU Thread Block", font_size=18).next_to(block_bg, UP, buff=0.2)

        threads = VGroup()
        for j in range(3):
            for i in range(3):
                thread_cell = RoundedRectangle(
                    width=1.6,
                    height=1.6,
                    corner_radius=0.1,
                    stroke_color=WHITE,
                    fill_color=DARK_GRAY,
                    fill_opacity=0.8
                )
                thread_text = Text(f"thread_idx=({j},{i})", font_size=14, color=YELLOW)
                valid_text = Text("if i,j < size", font_size=10, color=GREEN_A if (i < 2 and j < 2) else RED)
                thread_info = VGroup(thread_text, valid_text).arrange(DOWN, buff=0.05)
                thread_cell.add(thread_info)
                threads.add(thread_cell)

        threads.arrange_in_grid(rows=3, cols=3, buff=0.2)
        threads.move_to(block_bg)

        block_group = VGroup(block_bg, block_label, threads)
        block_group.move_to(ORIGIN)

        # Output matrix - positioned on the right
        output_matrix = VGroup()
        output_bg = Rectangle(
            width=array_scale * 2 + 1.0,
            height=array_scale * 2 + 1.0,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )

        for j in range(2):
            for i in range(2):
                cell = Square(side_length=array_scale + 0.1, stroke_width=1)
                index_text = Text(f"out[{j}][{i}]", font_size=10, color=YELLOW)
                cell.add(index_text)
                output_matrix.add(cell)
        output_matrix.arrange_in_grid(rows=2, cols=2, buff=0)
        output_matrix.move_to(output_bg)
        output_group = VGroup(output_bg, output_matrix)
        output_label = Text("Output Matrix (2×2)", font_size=18).next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)
        output_group.to_edge(RIGHT, buff=1.0)

        # Animations
        self.play(Write(input_label_a), Write(input_label_b))
        self.play(
            Create(input_bg_a), Create(input_vector_a),
            Create(input_bg_b), Create(input_vector_b),
            run_time=1.5
        )

        self.play(Write(block_label))
        self.play(Create(block_bg), Create(threads), run_time=1.5)

        self.play(Write(output_label))
        self.play(Create(output_bg), Create(output_matrix), run_time=1.5)

        # Create arrows flowing left to right
        arrows = VGroup()
        for j in range(3):
            for i in range(3):
                if i < 2 and j < 2:
                    # Arrow from input vector A to thread
                    start_a = input_vector_a[i].get_right()
                    end = threads[j * 3 + i].get_left()
                    arrow1 = Arrow(
                        start_a, end,
                        buff=0.2,
                        color=BLUE_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)

                    # Arrow from input vector B to thread
                    start_b = input_vector_b[j].get_right()
                    arrow2 = Arrow(
                        start_b, end,
                        buff=0.2,
                        color=BLUE_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)

                    # Arrow from thread to output
                    start = threads[j * 3 + i].get_right()
                    end = output_matrix[j * 2 + i].get_left()
                    arrow3 = Arrow(
                        start, end,
                        buff=0.2,
                        color=GREEN_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)

                    arrows.add(arrow1, arrow2, arrow3)
                else:
                    # Add X marks for invalid threads
                    warning = Text("×", font_size=36, color=RED).move_to(threads[j * 3 + i])
                    arrows.add(warning)

        self.play(FadeIn(arrows), run_time=0.3)
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_05_viz"
    }):
        scene = Puzzle05Visualization()
        scene.render()
