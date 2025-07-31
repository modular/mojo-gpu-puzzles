from manim import *

class MatmulVisualization(Scene):
    def construct(self):
        array_scale = 0.5

        # Add title at the top
        title = Text("Matrix Multiplication", font_size=32, color=YELLOW)
        title.to_edge(UP, buff=0.5)

        # Play title animation first
        self.play(Write(title))
        self.wait(0.5)

        # Matrix A (4x3)
        matrix_a_bg = Rectangle(
            width=array_scale * 4,
            height=array_scale * 5,
            stroke_color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.2
        )

        # Create A matrix cells
        matrix_a_cells = VGroup()
        for i in range(4):
            row = VGroup()
            for j in range(3):
                cell = Square(side_length=array_scale, stroke_width=1)
                index_text = Text(f"a[{i}][{j}]", font_size=10, color=YELLOW)
                cell.add(index_text)
                row.add(cell)
            row.arrange(RIGHT, buff=0.1)
            matrix_a_cells.add(row)

        matrix_a_cells.arrange(DOWN, buff=0.1)
        matrix_a_cells.move_to(matrix_a_bg)

        matrix_a_group = VGroup(matrix_a_bg, matrix_a_cells)
        matrix_a_label = Text("Matrix A (4×3)", font_size=18)
        matrix_a_label.next_to(matrix_a_group, UP, buff=0.2)
        matrix_a = VGroup(matrix_a_label, matrix_a_group)
        matrix_a.to_edge(LEFT, buff=2)
        matrix_a.to_edge(UP, buff=2)

        # Matrix B (3x4)
        matrix_b_bg = Rectangle(
            width=array_scale * 5,
            height=array_scale * 4,
            stroke_color=GREEN_D,
            fill_color=GREEN_E,
            fill_opacity=0.2
        )

        # Create B matrix cells
        matrix_b_cells = VGroup()
        for i in range(3):
            row = VGroup()
            for j in range(4):
                cell = Square(side_length=array_scale, stroke_width=1)
                index_text = Text(f"b[{i}][{j}]", font_size=10, color=YELLOW)
                cell.add(index_text)
                row.add(cell)
            row.arrange(RIGHT, buff=0.1)
            matrix_b_cells.add(row)

        matrix_b_cells.arrange(DOWN, buff=0.1)
        matrix_b_cells.move_to(matrix_b_bg)

        matrix_b_group = VGroup(matrix_b_bg, matrix_b_cells)
        matrix_b_label = Text("Matrix B (3×4)", font_size=18)
        matrix_b_label.next_to(matrix_b_group, UP, buff=0.2)
        matrix_b = VGroup(matrix_b_label, matrix_b_group)

        # Add multiplication symbol between A and B
        mult_symbol = Text("×", font_size=36, color=WHITE)
        mult_symbol.next_to(matrix_a, RIGHT, buff=0.8)

        # Position B after multiplication symbol
        matrix_b.next_to(mult_symbol, RIGHT, buff=0.8)

        # Add equals symbol between B and C
        equals_symbol = Text("=", font_size=36, color=WHITE)
        equals_symbol.next_to(matrix_b, RIGHT, buff=0.8)

        # Result Matrix C (4x4) - Create this BEFORE using its components
        matrix_c_bg = Rectangle(
            width=array_scale * 5,
            height=array_scale * 5,
            stroke_color=PURPLE_D,
            fill_color=PURPLE_E,
            fill_opacity=0.2
        )

        # Create C matrix cells
        matrix_c_cells = VGroup()
        for i in range(4):
            row = VGroup()
            for j in range(4):
                cell = Square(side_length=array_scale, stroke_width=1)
                index_text = Text(f"c[{i}][{j}]", font_size=10, color=YELLOW)
                cell.add(index_text)
                row.add(cell)
            row.arrange(RIGHT, buff=0.1)
            matrix_c_cells.add(row)

        matrix_c_cells.arrange(DOWN, buff=0.1)
        matrix_c_cells.move_to(matrix_c_bg)

        matrix_c_group = VGroup(matrix_c_bg, matrix_c_cells)
        matrix_c_label = Text("Matrix C (4×4)", font_size=18)
        matrix_c_label.next_to(matrix_c_group, UP, buff=0.2)

        # Now create the complete matrix C group
        matrix_c = VGroup(matrix_c_label, matrix_c_group)
        matrix_c.next_to(equals_symbol, RIGHT, buff=0.8)

        # Adjust vertical positions - shift everything up by adjusting the DOWN shift
        matrix_a.shift(DOWN * 0.3)
        matrix_b.shift(DOWN * 0.3)
        matrix_c.shift(DOWN * 0.3)
        mult_symbol.shift(DOWN * 0.3)
        equals_symbol.shift(DOWN * 0.3)

        # Initial display with operation symbols
        self.play(
            Write(matrix_a),
            Write(mult_symbol),
            Write(matrix_b),
            Write(equals_symbol),
            Write(matrix_c)
        )
        self.wait(0.5)

        # Demonstrate computation for C[1,2] (second row, third column)
        i, j = 1, 2  # Example indices

        # Highlight row i of matrix A
        row_highlight = Rectangle(
            width=array_scale * 3 + 0.2,
            height=array_scale + 0.1,
            stroke_color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.2
        ).move_to(matrix_a_cells[i])

        # Highlight column j of matrix B
        col_highlight = Rectangle(
            width=array_scale + 0.1,
            height=array_scale * 3 + 0.2,
            stroke_color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.2
        ).move_to(VGroup(*[matrix_b_cells[k][j] for k in range(3)]))

        # Highlight target cell in C
        target_highlight = Square(
            side_length=array_scale + 0.1,
            stroke_color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.2
        ).move_to(matrix_c_cells[i][j])

        self.play(
            FadeIn(row_highlight),
            FadeIn(col_highlight),
            FadeIn(target_highlight)
        )

        # Show dot product computation in a horizontal line
        dot_product_terms = VGroup()
        plus_signs = VGroup()
        arrows = VGroup()

        # Adjust dot product line position to account for new spacing
        dot_product_line_center = matrix_b.get_center() + DOWN * 3

        for k in range(3):
            # Create multiplication terms
            term = Text(f"a[{i}][{k}]×b[{k}][{j}]", font_size=18, color=WHITE)

            if k == 0:
                term.move_to(dot_product_line_center + LEFT * 3)
            else:
                # Add plus sign before the term
                plus = Text("+", font_size=18, color=WHITE)
                plus.next_to(dot_product_terms[-1], RIGHT, buff=0.2)
                plus_signs.add(plus)
                # Position term after plus sign
                term.next_to(plus, RIGHT, buff=0.2)

            dot_product_terms.add(term)

            # Create smaller arrows from matrix elements to terms
            arrow1 = Arrow(
                matrix_a_cells[i][k].get_center(),
                term.get_top(),
                buff=0.1,
                color=BLUE_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15
            ).set_opacity(0.6)

            arrow2 = Arrow(
                matrix_b_cells[k][j].get_center(),
                term.get_top(),
                buff=0.1,
                color=GREEN_C,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15
            ).set_opacity(0.6)

            arrows.add(arrow1, arrow2)

            # Animate each term and its arrows
            if k > 0:
                self.play(
                    Create(arrow1),
                    Create(arrow2),
                    Write(term),
                    Write(plus_signs[-1])
                )
            else:
                self.play(
                    Create(arrow1),
                    Create(arrow2),
                    Write(term)
                )
            self.wait(0.3)

        # Show equals sign and result
        equals_sign = Text("=", font_size=18, color=WHITE)
        equals_sign.next_to(dot_product_terms[-1], RIGHT, buff=0.2)

        result_text = Text("c[1][2]", font_size=20, color=YELLOW)
        result_text.next_to(equals_sign, RIGHT, buff=0.2)

        # Smaller result arrow
        result_arrow = Arrow(
            result_text.get_right(),
            target_highlight.get_center(),
            buff=0.1,
            color=PURPLE_C,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15
        ).set_opacity(0.6)

        self.play(
            Write(equals_sign),
            Write(result_text),
            Create(result_arrow)
        )
        self.wait(0.5)

        # Cleanup
        self.play(
            FadeOut(arrows),
            FadeOut(dot_product_terms),
            FadeOut(plus_signs),
            FadeOut(equals_sign),
            FadeOut(result_text),
            FadeOut(result_arrow),
            FadeOut(row_highlight),
            FadeOut(col_highlight),
            FadeOut(target_highlight)
        )

        # Show text about repeating for all elements
        final_text = Text("Repeat for all elements of C", font_size=18, color=YELLOW)
        final_text.next_to(matrix_b, DOWN, buff=1)
        self.play(Write(final_text))
        self.wait(60)

        # At the end, fade out title and subtitle with everything else
        self.play(
            FadeOut(title),
            *[FadeOut(mob) for mob in self.mobjects]
        )

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "puzzle_16_viz"
    }):
        scene = MatmulVisualization()
        scene.render()
