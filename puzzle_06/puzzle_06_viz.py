from manim import *

class Puzzle06Visualization(Scene):
    def construct(self):
        array_scale = 0.5  # Reduced scale
        thread_scale = 0.4

        # Input array (size 9) - positioned at the top, centered
        input_array = VGroup()
        input_bg = Rectangle(
            width=array_scale * 9 + 1.0,
            height=array_scale + 0.6,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        ).shift(UP * 2.5)

        for i in range(9):
            cell = Square(side_length=array_scale, stroke_width=1)
            index_text = Text(f"a[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_array.add(cell)
        input_array.arrange(RIGHT, buff=0)
        input_array.move_to(input_bg)
        input_group = VGroup(input_bg, input_array)
        input_label = Text("Input Array (size=9)", font_size=18).next_to(input_group, UP, buff=0.2)
        input_group = VGroup(input_label, input_group)

        # GPU Grid - keep width but adjust internal spacing
        grid_bg = Rectangle(
            width=14,
            height=1.8,
            stroke_color=GOLD_D,
            fill_color=DARK_GRAY,
            fill_opacity=0.1
        )
        grid_label = Text("GPU (gridDim.x=3, gridDim.y=1) • (blockDim.x=4, blockDim.y=1)", font_size=18).next_to(grid_bg, UP, buff=0.2)

        # Create Thread Blocks with tighter spacing
        thread_blocks = VGroup()
        block_separators = VGroup()

        for block_idx_x in range(3):
            block_bg = Rectangle(
                width=4.5,
                height=1.2,
                stroke_color=PURPLE_D,
                fill_color=DARK_GRAY,
                fill_opacity=0.2
            )

            # Add vertical separator after each block (except last)
            if block_idx_x < 2:
                separator = Line(
                    start=grid_bg.get_top() + DOWN * 0.3,
                    end=grid_bg.get_bottom() + UP * 0.3,
                    stroke_color=BLUE_D
                )
                block_separators.add(separator)

            block_label = Text(f"block_idx.x={block_idx_x}", font_size=12).next_to(block_bg, UP, buff=0.1)

            threads = VGroup()
            for i in range(4):
                global_idx = block_idx_x * 4 + i
                thread_cell = RoundedRectangle(
                    width=1.1,
                    height=0.9,
                    corner_radius=0.1,
                    stroke_color=WHITE,
                    fill_color=DARK_GRAY,
                    fill_opacity=0.8
                )
                thread_text = Text(
                    f"thread_idx.x={i}",
                    font_size=9,
                    color=YELLOW if global_idx < 9 else RED
                )
                thread_cell.add(thread_text)
                threads.add(thread_cell)

            threads.arrange(RIGHT, buff=0)
            threads.move_to(block_bg)

            block_group = VGroup(block_bg, block_label, threads)
            thread_blocks.add(block_group)

        # Increase spacing between blocks
        thread_blocks.arrange(RIGHT, buff=0.1)

        # Center in grid with margin
        thread_blocks.move_to(grid_bg)
        thread_blocks.shift(UP * 0.1)

        # Adjust separator positions to match new block spacing
        for i, separator in enumerate(block_separators):
            x_pos = thread_blocks[i].get_right()[0] + 0.15  # Increased from 0.1 to 0.15
            separator.move_to([x_pos, grid_bg.get_center()[1], 0])

        grid_group = VGroup(grid_bg, grid_label, thread_blocks, block_separators)
        grid_group.move_to(ORIGIN)
        grid_group.shift(UP * 0.3)

        # Output array - adjusted cell size
        output_array = VGroup()
        output_bg = Rectangle(
            width=array_scale * 9 + 1.5,  # Increased padding from 1.0 to 2.0
            height=array_scale + 1,  # Increased height padding
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        ).shift(DOWN * 2.5)

        for i in range(9):
            cell = Square(
                side_length=array_scale * 1.2,
                stroke_width=1
            )
            index_text = Text(f"out[{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            output_array.add(cell)
        output_array.arrange(RIGHT, buff=0)
        output_array.move_to(output_bg)
        output_group = VGroup(output_bg, output_array)
        output_label = Text("Output Array (size=9)", font_size=18).next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)

        # Animations
        self.play(Write(input_label))
        self.play(Create(input_bg), Create(input_array), run_time=1.5)

        self.play(Write(grid_label))
        self.play(
            Create(grid_bg),
            Create(thread_blocks),
            run_time=1.5
        )

        self.play(Write(output_label))
        self.play(Create(output_bg), Create(output_array), run_time=1.5)

        # Create arrows flowing top to bottom
        arrows = VGroup()
        for block_idx_x in range(3):
            for thread_idx_x in range(4):
                global_idx = block_idx_x * 4 + thread_idx_x
                if global_idx < 9:  # Only create arrows for valid threads
                    # Arrow from input to thread
                    start = input_array[global_idx].get_bottom()
                    end = thread_blocks[block_idx_x][2][thread_idx_x].get_top()
                    arrow1 = Arrow(
                        start, end,
                        buff=0.2,
                        color=BLUE_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)

                    # Arrow from thread to output
                    start = thread_blocks[block_idx_x][2][thread_idx_x].get_bottom()
                    end = output_array[global_idx].get_top()
                    arrow2 = Arrow(
                        start, end,
                        buff=0.2,
                        color=GREEN_C,
                        stroke_width=2,
                        max_tip_length_to_length_ratio=0.2
                    ).set_opacity(0.6)

                    arrows.add(arrow1, arrow2)

        self.play(FadeIn(arrows), run_time=0.3)
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_06_viz"
    }):
        scene = Puzzle06Visualization()
        scene.render()
