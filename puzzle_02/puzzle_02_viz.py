from manim import *

class Puzzle02Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # First input array (a)
        input_array_a = VGroup()
        input_bg_a = Rectangle(
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
            input_array_a.add(cell)
        input_array_a.arrange(RIGHT, buff=0)
        input_array_a.move_to(input_bg_a)
        input_group_a = VGroup(input_bg_a, input_array_a)
        input_label_a = Text("Input Array A (size=4)", font_size=18).next_to(input_group_a, UP, buff=0.2)
        input_group_a = VGroup(input_label_a, input_group_a)

        # Second input array (b)
        input_array_b = VGroup()
        input_bg_b = Rectangle(
            width=array_scale * 4 + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )

        for i in range(4):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"b[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_array_b.add(cell)
        input_array_b.arrange(RIGHT, buff=0)
        input_array_b.move_to(input_bg_b)
        input_group_b = VGroup(input_bg_b, input_array_b)
        input_label_b = Text("Input Array B (size=4)", font_size=18).next_to(input_group_b, UP, buff=0.2)
        input_group_b = VGroup(input_label_b, input_group_b)

        # Arrange input arrays side by side with consistent spacing
        input_arrays = VGroup(input_group_a, input_group_b).arrange(RIGHT, buff=1.5)
        input_arrays.shift(UP * 2.5)  # Move both arrays up together after arranging

        # GPU Thread Block in the middle
        block_bg = Rectangle(
            width=9,
            height=2.4,
            stroke_color=GOLD_D,
            fill_color=DARK_GRAY,
            fill_opacity=0.1
        )
        block_label = Text("GPU Parallel Threads in a Block", font_size=18).next_to(block_bg, UP, buff=0.2)

        threads = VGroup()
        for i in range(4):
            thread_cell = RoundedRectangle(
                width=1.8,
                height=1.2,
                corner_radius=0.1,
                stroke_color=WHITE,
                fill_color=DARK_GRAY,
                fill_opacity=0.8
            )
            thread_text = Text(f"thread_idx.x={i}", font_size=16, color=YELLOW)
            thread_cell.add(thread_text)
            threads.add(thread_cell)

        threads.arrange(RIGHT, buff=0.3)
        threads.move_to(block_bg)
        threads.shift(UP * 0.1)

        block_group = VGroup(block_bg, block_label, threads)
        block_group.move_to(ORIGIN)  # Center the block group

        # Output array at the bottom
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

        output_group.shift(DOWN * 2.5)  # Move output array down after creation

        # Animations
        self.play(Write(input_label_a), Write(input_label_b))
        self.play(
            Create(input_bg_a),
            Create(input_array_a),
            Create(input_bg_b),
            Create(input_array_b),
            run_time=1.5
        )

        self.play(Write(block_label))
        self.play(Create(block_bg), Create(threads), run_time=1.5)

        self.play(Write(output_label))
        self.play(Create(output_bg), Create(output_array), run_time=1.5)

        # Create arrows flowing top to bottom
        arrows = VGroup()
        for i in range(4):
            # Arrows from first input array to thread
            start_a = input_array_a[i].get_bottom()
            end = threads[i].get_top()
            arrow1 = Arrow(
                start_a, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # Arrows from second input array to thread
            start_b = input_array_b[i].get_bottom()
            arrow2 = Arrow(
                start_b, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # Arrow from thread to output
            start = threads[i].get_bottom()
            end = output_array[i].get_top()
            arrow3 = Arrow(
                start, end,
                buff=0.2,
                color=GREEN_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            arrows.add(arrow1, arrow2, arrow3)

        self.play(FadeIn(arrows), run_time=0.3)
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_02_viz"
    }):
        scene = Puzzle02Visualization()
        scene.render()
