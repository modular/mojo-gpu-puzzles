from manim import *

class ThreadIndexingConvention(Scene):
    def construct(self):
        # Increase base scale
        array_scale = 1.0

        # Create matrix grid first
        matrix_bg = Rectangle(
            width=array_scale * 3 + 1.0,
            height=array_scale * 3 + 1.0,
            stroke_color=BLUE_D,
            stroke_width=5,
            fill_color=BLUE_E,
            fill_opacity=0.2
        ).move_to(ORIGIN).shift(LEFT * 2)  # Shifted more left to make room

        # Create axes first (so they go behind the matrix)
        # Y-axis along left edge and extending down
        y_axis = VGroup()
        y_line = Line(
            matrix_bg.get_corner(UL),
            matrix_bg.get_corner(DL) + DOWN,
            color=GREEN_D,
            stroke_width=3
        )
        y_arrow = Arrow(
            y_line.get_end() - DOWN * 0.5,
            y_line.get_end(),
            color=GREEN_D,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.5,
            tip_length=0.4,
            buff=0
        )

        # Y-axis label (vertical)
        y_text = Text("thread_idx.y → rows", font_size=24, color=GREEN_D)
        y_text.rotate(90 * DEGREES)  # Make vertical
        y_text.next_to(matrix_bg, LEFT, buff=0.5)  # Position left of matrix
        y_axis.add(y_line, y_arrow, y_text)

        # Now create matrix squares
        matrix_squares = VGroup()
        for y in range(3):
            for x in range(3):
                cell = Square(
                    side_length=array_scale,
                    stroke_width=2,
                    stroke_color=WHITE
                )
                pos_label = Text(
                    f"({y},{x})",
                    font_size=24,
                    color=YELLOW
                )
                cell.add(pos_label)
                matrix_squares.add(cell)

        matrix_squares.arrange_in_grid(rows=3, cols=3, buff=0)
        matrix_squares.move_to(matrix_bg)
        matrix_group = VGroup(matrix_bg, matrix_squares)

        # X-axis along top edge and extending right
        x_axis = VGroup()
        x_line = Line(
            matrix_bg.get_corner(UL),
            matrix_bg.get_corner(UR) + RIGHT,
            color=BLUE_D,
            stroke_width=5
        )
        x_arrow = Arrow(
            x_line.get_end() - RIGHT * 0.5,
            x_line.get_end(),
            color=BLUE_D,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.5,
            tip_length=0.4,
            buff=0
        )

        # X-axis label (horizontal)
        x_text = Text("thread_idx.x → columns", font_size=24, color=BLUE_D)
        x_text.next_to(matrix_bg, UP, buff=0.2)  # Position above matrix
        x_axis.add(x_line, x_arrow, x_text)

        # Title and explanation
        title = Text(
            "2D Matrix Coordinates with Thread Indices",
            font_size=34
        ).to_edge(UP, buff=0.5)

        explanation = VGroup(
            Text("• (0,0) starts at top-left", font_size=24, color=YELLOW),
            Text("• thread_idx.x increases rightward →", font_size=24, color=BLUE_D),
            Text("• thread_idx.y increases downward ↓", font_size=24, color=GREEN_D)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        explanation.to_edge(RIGHT, buff=0.8)

        # Animation sequence - note y_axis is added first
        self.play(Write(title))

        # Add y-axis first so it goes behind matrix
        self.play(
            Create(y_line),
            Create(y_arrow),
            run_time=1
        )

        self.play(
            Create(matrix_bg),
            Create(matrix_squares),
            run_time=1.5
        )

        self.play(
            Create(x_line),
            Create(x_arrow),
            run_time=1
        )

        self.play(
            Write(x_text),
            Write(y_text),
            run_time=1
        )

        self.play(Write(explanation), run_time=1)

        # Highlight (0,0) position
        highlight = matrix_squares[0].copy().set_stroke(color=YELLOW, width=4)
        self.play(FadeIn(highlight), run_time=0.5)

        self.wait(60)

if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "thread_indexing_viz"
    }):
        scene = ThreadIndexingConvention()
        scene.render()
