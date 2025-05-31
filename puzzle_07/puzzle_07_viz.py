from manim import *

# Constants matching p07.mojo
SIZE = 5
THREADS_PER_BLOCK = (3, 3)

class Puzzle07Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input 5x5 matrix (showing corners with ellipsis)
        input_matrix = VGroup()
        input_bg = Rectangle(
            width=array_scale * 5 + 1.0,
            height=array_scale * 5 + 1.0,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )

        # Create corner cells and ellipsis
        positions = [(0,0), (0,4), (4,0), (4,4)]
        for j, i in positions:
            cell = Square(side_length=array_scale + 0.1, stroke_width=1)
            index_text = Text(f"a[{j}][{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            input_matrix.add(cell)

        top_row = VGroup(input_matrix[0], Text("...", color=YELLOW), input_matrix[1]).arrange(RIGHT, buff=0.3)
        bottom_row = VGroup(input_matrix[2], Text("...", color=YELLOW), input_matrix[3]).arrange(RIGHT, buff=0.3)
        vertical_dots = Text("⋮", color=YELLOW, font_size=24)
        matrix_group = VGroup(top_row, vertical_dots, bottom_row).arrange(DOWN, buff=0.3)
        matrix_group.move_to(input_bg)
        input_group = VGroup(input_bg, matrix_group)
        input_label = Text("Input Matrix (5×5)", font_size=18).next_to(input_group, UP, buff=0.2)
        input_group = VGroup(input_label, input_group)
        input_group.to_edge(LEFT, buff=0.5)

        # GPU Grid with Thread Blocks - make it smaller
        grid_bg = Rectangle(
            width=5.5,  # Reduced from 7
            height=5.5,  # Reduced from 7
            stroke_color=GOLD_D,
            fill_color=DARK_GRAY,
            fill_opacity=0.1
        )
        grid_label = Text("GPU (gridDim.x=2, gridDim.y=2) • (blockDim.x=3, blockDim.y=3)", font_size=18).next_to(grid_bg, UP, buff=0.2)

        # Create 2x2 blocks, each showing 3x3 threads - make blocks smaller
        blocks = VGroup()
        for block_j in range(2):
            for block_i in range(2):
                block = VGroup()
                block_bg = Rectangle(
                    width=2.4,  # Reduced from 3.0
                    height=2.4,  # Reduced from 3.0
                    stroke_color=PURPLE_D,
                    fill_color=DARK_GRAY,
                    fill_opacity=0.2
                )

                # Show corner threads with ellipsis
                threads = VGroup()
                thread_positions = [(0,0), (0,2), (2,0), (2,2)]
                for tj, ti in thread_positions:
                    thread_cell = RoundedRectangle(
                        width=0.7,  # Reduced from 0.8
                        height=0.7,  # Reduced from 0.8
                        corner_radius=0.1,
                        stroke_color=WHITE,
                        fill_color=DARK_GRAY,
                        fill_opacity=0.8
                    )
                    thread_text = Text(f"T({tj},{ti})", font_size=10, color=YELLOW)  # Reduced font size
                    thread_cell.add(thread_text)
                    threads.add(thread_cell)

                # Arrange threads with ellipsis
                top_threads = VGroup(threads[0], Text("...", color=YELLOW), threads[1]).arrange(RIGHT, buff=0.2)
                bottom_threads = VGroup(threads[2], Text("...", color=YELLOW), threads[3]).arrange(RIGHT, buff=0.2)
                thread_dots = Text("⋮", color=YELLOW, font_size=24)
                thread_group = VGroup(top_threads, thread_dots, bottom_threads).arrange(DOWN, buff=0.2)
                thread_group.move_to(block_bg)

                block.add(block_bg, thread_group)
                blocks.add(block)

        blocks.arrange_in_grid(rows=2, cols=2, buff=0.3)  # Reduced buffer from 0.4
        blocks.move_to(grid_bg)

        grid_group = VGroup(grid_bg, grid_label, blocks)
        grid_group.move_to(ORIGIN)

        # Output matrix (showing corners with ellipsis)
        output_matrix = VGroup()
        output_bg = Rectangle(
            width=array_scale * 5 + 1.0,
            height=array_scale * 5 + 1.0,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )

        for j, i in positions:
            cell = Square(side_length=array_scale + 0.2, stroke_width=1)
            index_text = Text(f"out[{j}][{i}]", font_size=10, color=YELLOW)
            cell.add(index_text)
            output_matrix.add(cell)

        out_top_row = VGroup(output_matrix[0], Text("...", color=YELLOW), output_matrix[1]).arrange(RIGHT, buff=0.3)
        out_bottom_row = VGroup(output_matrix[2], Text("...", color=YELLOW), output_matrix[3]).arrange(RIGHT, buff=0.3)
        out_dots = Text("⋮", color=YELLOW, font_size=24)
        output_group = VGroup(out_top_row, out_dots, out_bottom_row).arrange(DOWN, buff=0.3)
        output_group.move_to(output_bg)
        matrix_output_group = VGroup(output_bg, output_group)
        output_label = Text("Output Matrix (5×5)", font_size=18).next_to(matrix_output_group, UP, buff=0.2)
        output_group = VGroup(output_label, matrix_output_group)
        output_group.to_edge(RIGHT, buff=0.5)

        # Create crosses before animations
        crosses = VGroup()
        for block_j in range(2):
            for block_i in range(2):
                block = blocks[block_j * 2 + block_i]
                thread_group = block[1]  # Get thread group from block
                for tj, ti in thread_positions:
                    # Calculate global indices
                    global_i = THREADS_PER_BLOCK[0] * block_i + ti
                    global_j = THREADS_PER_BLOCK[1] * block_j + tj

                    # If thread is inactive (outside 5x5 matrix)
                    if global_i >= SIZE or global_j >= SIZE:
                        # Find the correct thread cell
                        if tj == 0 and ti == 0:
                            thread_cell = thread_group[0][0]  # Top-left thread
                        elif tj == 0 and ti == 2:
                            thread_cell = thread_group[0][2]  # Top-right thread
                        elif tj == 2 and ti == 0:
                            thread_cell = thread_group[2][0]  # Bottom-left thread
                        elif tj == 2 and ti == 2:
                            thread_cell = thread_group[2][2]  # Bottom-right thread

                        # Create red cross
                        cross = VGroup(
                            Line(
                                start=thread_cell.get_corner(UL) + 0.1 * RIGHT + 0.1 * DOWN,
                                end=thread_cell.get_corner(DR) + (-0.1) * RIGHT + (-0.1) * DOWN,
                                color=RED,
                                stroke_width=2
                            ),
                            Line(
                                start=thread_cell.get_corner(DL) + 0.1 * RIGHT + (-0.1) * DOWN,
                                end=thread_cell.get_corner(UR) + (-0.1) * RIGHT + 0.1 * DOWN,
                                color=RED,
                                stroke_width=2
                            )
                        )
                        crosses.add(cross)

        # Add calculation labels for the first block and thread
        calc_labels = VGroup()

        # Block index labels (for top-left block)
        block_idx_label = Text("block_idx = (0,0)", font_size=13, color=WHITE)
        block_dim_label = Text("block_dim = (3,3)", font_size=13, color=WHITE)

        # Arrange labels side by side
        block_labels = VGroup(block_idx_label, block_dim_label).arrange(RIGHT, buff=0.1)
        block_labels.next_to(blocks[0], UP, buff=0).shift(0.1 * RIGHT)

        # Thread index labels (for top-left thread of first block)
        thread_idx_label = Text("thread_idx = (0,0)", font_size=12, color=WHITE)
        thread_idx_label.next_to(blocks[0][1][0], LEFT, buff=0.1)

        calc_labels.add(block_labels, thread_idx_label)

        # Animations
        self.play(Write(input_label))
        self.play(Create(input_bg), Create(matrix_group), run_time=1.5)

        self.play(Write(grid_label))
        self.play(Create(grid_bg), Create(blocks), Create(crosses), run_time=1.5)
        self.play(Write(calc_labels), run_time=1.0)

        self.play(Write(output_label))
        self.play(Create(output_group), run_time=1.5)

        # Create sample arrows for visible elements
        arrows = VGroup()
        # Sample arrows from first block
        start = input_matrix[0].get_right()
        end = blocks[0][1][0].get_left()  # First thread in first block
        arrow1 = Arrow(
            start, end,
            buff=0.2,
            color=BLUE_C,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        ).set_opacity(0.6)

        start = blocks[0][1][0].get_right()
        end = output_matrix[0].get_left()
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
        "output_file": "puzzle_07_viz"
    }):
        scene = Puzzle07Visualization()
        scene.render()
