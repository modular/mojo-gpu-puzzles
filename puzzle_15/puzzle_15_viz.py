from manim import *

BATCH = 4
SIZE = 6
TPB = 8

class Puzzle15Visualization(Scene):
    def construct(self):
        array_scale = 0.5
        thread_scale = 0.4

        # Input matrix - show corners with ellipsis
        input_matrix = VGroup()
        matrix_bg = Rectangle(
            width=array_scale * 3 + 1.0,  # Reduced width to show only corners
            height=array_scale * BATCH + 1.0,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )

        # Create corner cells and ellipsis
        matrix_cells = VGroup()
        for batch in range(BATCH):
            row = VGroup()
            # First element
            cell1 = Square(side_length=array_scale, stroke_width=1)
            index_text1 = Text(f"a[{batch}][0]", font_size=10, color=YELLOW)
            cell1.add(index_text1)
            # Last element
            cell2 = Square(side_length=array_scale, stroke_width=1)
            index_text2 = Text(f"a[{batch}][{SIZE-1}]", font_size=10, color=YELLOW)
            cell2.add(index_text2)
            # Add horizontal dots
            row.add(cell1, Text("...", color=YELLOW), cell2)
            row.arrange(RIGHT, buff=0.2)
            matrix_cells.add(row)

        matrix_cells.arrange(DOWN, buff=0.2)
        matrix_cells.move_to(matrix_bg)

        matrix_group = VGroup(matrix_bg, matrix_cells)
        matrix_label = Text("Input Matrix (4×6)", font_size=18)
        matrix_label.next_to(matrix_group, UP, buff=0.2)
        input_group = VGroup(matrix_label, matrix_group)
        input_group.to_edge(LEFT, buff=0.5)

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
        shared_label = Text("Shared Memory Cache (TPB=8)", font_size=14, color=WHITE)
        parallel_text = Text("Parallel row reduction", font_size=14, color=YELLOW)
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

        # Create initial block label and grid label
        block_label = Text("Block(0,0)", font_size=14, color=WHITE)
        block_label.next_to(block_bg, UP, buff=0.2)

        grid_label = Text("Grid(1×4) • Block(8×1)", font_size=14, color=WHITE)
        grid_label.next_to(block_label, UP, buff=0.2)

        # Add labels to a label group that will move with the block
        labels = VGroup(block_label, grid_label)

        block = VGroup(
            block_bg,
            threads,
            shared_label_group,
            shared_mem,
            shared_cells,
        )
        block.move_to(ORIGIN)

        # Output array - show as vertical array with dots
        output_bg = Rectangle(
            width=array_scale + 1.0,
            height=array_scale * 3 + 1.0,  # Reduced height to show only corners
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )

        output_array = VGroup()
        # First element
        cell1 = Square(side_length=array_scale, stroke_width=1)
        cell1.add(Text("out[0]", font_size=10, color=YELLOW))
        # Vertical dots
        dots = Text("⋮", color=YELLOW, font_size=24)
        # Last element
        cell2 = Square(side_length=array_scale, stroke_width=1)
        cell2.add(Text(f"out[{BATCH-1}]", font_size=10, color=YELLOW))

        output_array.add(cell1, dots, cell2)
        output_array.arrange(DOWN, buff=0.2)
        output_array.move_to(output_bg)

        output_group = VGroup(output_bg, output_array)
        output_label = Text("Output Array (size=4)", font_size=18)
        output_label.next_to(output_group, UP, buff=0.2)
        output_group = VGroup(output_label, output_group)
        output_group.to_edge(RIGHT, buff=0.5)

        # Initial animations
        self.play(Write(input_group))
        self.play(Create(block), Create(labels))
        self.play(Create(output_group))

        # Show multiple row processing
        for batch in range(2):  # Show first two rows to demonstrate pattern
            # Create new labels for the current batch
            new_block_label = Text(f"Block(0,{batch})", font_size=14, color=WHITE)
            new_block_label.next_to(block_bg, UP, buff=0.2)
            new_grid_label = Text("Grid(1×4) • Block(8×1)", font_size=14, color=WHITE)
            new_grid_label.next_to(new_block_label, UP, buff=0.2)
            new_labels = VGroup(new_block_label, new_grid_label)

            # Move block and transform labels together
            self.play(
                block.animate.move_to(block.get_center() + DOWN * batch * array_scale * 2),
                Transform(labels, new_labels.move_to(new_labels.get_center() + DOWN * batch * array_scale * 2))
            )

            # Show data loading from current row to shared memory
            load_arrows = VGroup()
            # Arrow from first element
            start = matrix_cells[batch][0].get_center()
            end = shared_cells[0].get_top()
            arrow1 = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # Arrow from last element
            start = matrix_cells[batch][2].get_center()
            end = shared_cells[SIZE-1].get_top()
            arrow2 = Arrow(
                start, end,
                buff=0.2,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            # Add loading text showing the exact indexing
            load_text = Text(f"Loading: a[{batch}][0:{SIZE}]", font_size=14, color=YELLOW)
            load_text.next_to(shared_cells, UP, buff=0.2)

            load_arrows.add(arrow1, arrow2, load_text)

            self.play(FadeIn(load_arrows))
            self.wait(0.5)
            self.play(FadeOut(load_arrows))

            # Show parallel reduction with stride halving
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

                    # Curved arrows showing parallel reduction
                    curved_arrow = CurvedArrow(
                        start_point=shared_cells[i + stride].get_center(),
                        end_point=shared_cells[i].get_center(),
                        angle=-TAU/4,
                        color=GREEN_C,
                        stroke_width=2,
                        tip_length=0.2
                    )

                    # Operation text
                    midpoint = (shared_cells[i].get_center() + shared_cells[i + stride].get_center()) / 2
                    op_text = Text("+", font_size=32, color=GREEN_C, weight=BOLD)
                    op_text.move_to(midpoint)
                    op_text.shift(DOWN * 0.5)

                    arrows.add(VGroup(curved_arrow, op_text))

                # Show reduction step info
                active_text = Text(f"Parallel reduction: stride={stride}", font_size=14, color=YELLOW)
                active_text.next_to(shared_cells, DOWN, buff=0.5)

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

            # Show final output for this row
            output_arrow = Arrow(
                shared_cells[0].get_center(),
                output_array[batch].get_center(),
                buff=0.2,
                color=GREEN_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2
            ).set_opacity(0.6)

            output_text = Text(f"out[{batch}] = sum(a[{batch}][:])", font_size=14, color=GREEN_C)
            output_text.next_to(output_arrow, UP, buff=0.1)

            self.play(
                FadeIn(output_arrow),
                FadeIn(output_text)
            )
            self.wait(0.5)
            self.play(
                FadeOut(output_arrow),
                FadeOut(output_text)
            )

        # Show "..." to indicate remaining rows processed similarly
        continue_text = Text("Remaining rows are processed similarly ...", font_size=18, color=YELLOW)
        continue_text.next_to(block, DOWN, buff=0.3)
        self.play(Write(continue_text))
        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_15_viz"
    }):
        scene = Puzzle15Visualization()
        scene.render()
