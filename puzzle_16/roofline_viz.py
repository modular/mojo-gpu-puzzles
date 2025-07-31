from manim import *
import numpy as np

class RooflineModelVisualization(Scene):
    def construct(self):
        # Hardware parameters (NVIDIA A100)
        P_peak = 19500  # GFLOP/s
        B_peak = 1555   # GB/s
        I_critical = P_peak / B_peak  # 12.5 FLOP/B

        # Title - larger font
        title = Text("The Roofline Model", font_size=36, color=YELLOW, weight=BOLD)
        title.to_edge(UP, buff=0.5)

        # Set up the coordinate system (log-log scale)
        # Fixed: x starts from 0 (1 FLOP/B), y starts from 0 (1 GFLOP/s)
        axes = Axes(
            x_range=[0, 2.5, 0.5],
            y_range=[0, 5.5, 0.5],
            x_length=7,
            y_length=5,
            axis_config={"color": WHITE, "stroke_width": 2},
            x_axis_config={
                "numbers_to_include": [],
            },
            y_axis_config={
                "numbers_to_include": [],
            },
        )
        axes.shift(DOWN * 0.1 + LEFT * 0.8)

        # Axis labels - using Text instead of MathTex
        x_label = Text("log₁₀(I) where I [FLOP/B]", font_size=24, color=WHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.8)
        x_label.set_stroke(BLACK, width=3, background=True)

        y_label = Text("log₁₀(P) where P [GFLOP/s]", font_size=24, color=WHITE)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2).rotate(PI/2)
        y_label.set_stroke(BLACK, width=3, background=True)

        # Create custom tick labels for log scale - using Text instead of MathTex
        x_tick_labels = VGroup()

        # Major ticks (powers of 10 only)
        x_major_values = [1, 10, 100]
        x_major_positions = [0, 1, 2]

        for pos, val in zip(x_major_positions, x_major_values):
            main_label = Text(f"{val}", font_size=20, color=WHITE)
            main_label.set_stroke(BLACK, width=2, background=True)
            log_notation = Text(f"10^{int(np.log10(val))}", font_size=14, color=GRAY_B)
            log_notation.set_stroke(BLACK, width=2, background=True)

            main_label.next_to(axes.c2p(pos, 0), DOWN, buff=0.3)
            log_notation.next_to(main_label, DOWN, buff=0.1)

            x_tick_labels.add(main_label, log_notation)

        y_tick_labels = VGroup()

        # Major ticks (powers of 10 only)
        y_major_positions = [0, 1, 2, 3, 4, 5]
        y_major_values = [1, 10, 100, 1000, 10000, 100000]

        for pos, val in zip(y_major_positions, y_major_values):
            if val >= 1000:
                if val >= 100000:
                    main_label = Text("100K", font_size=20, color=WHITE)
                    log_notation = Text("10^5", font_size=14, color=GRAY_B)
                elif val >= 10000:
                    main_label = Text("10K", font_size=20, color=WHITE)
                    log_notation = Text("10^4", font_size=14, color=GRAY_B)
                else:
                    main_label = Text("1K", font_size=20, color=WHITE)
                    log_notation = Text("10^3", font_size=14, color=GRAY_B)
            else:
                main_label = Text(f"{val}", font_size=20, color=WHITE)
                log_notation = Text(f"10^{int(np.log10(val))}", font_size=14, color=GRAY_B)

            main_label.set_stroke(BLACK, width=2, background=True)
            log_notation.set_stroke(BLACK, width=2, background=True)

            main_label.next_to(axes.c2p(0, pos), LEFT, buff=0.4)
            log_notation.next_to(main_label, LEFT, buff=0.1)

            y_tick_labels.add(main_label, log_notation)

        # Step 1: Introduction
        self.play(Write(title))
        self.wait(0.5)

        # Step 2: Set up axes
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        self.play(
            Write(x_tick_labels),
            Write(y_tick_labels)
        )
        self.wait(1)

        # Step 3: Explanation
        explanation_group = VGroup()

        intro_text = Text("Performance is limited by:", font_size=22, color=WHITE, weight=BOLD)
        intro_text.to_edge(RIGHT, buff=0.5).shift(UP * 2.5)

        compute_bullet = Text("• Compute (Arithmetic Units)", font_size=18, color=BLUE_D)
        compute_bullet.next_to(intro_text, DOWN, aligned_edge=LEFT, buff=0.15)

        memory_bullet = Text("• Memory Bandwidth", font_size=18, color=RED_D)
        memory_bullet.next_to(compute_bullet, DOWN, aligned_edge=LEFT, buff=0.15)

        explanation_group.add(intro_text, compute_bullet, memory_bullet)

        self.play(Write(explanation_group))
        self.wait(2)

        # Step 4: Draw the compute roof (horizontal line)
        compute_roof_y = np.log10(P_peak)
        compute_roof = axes.get_horizontal_line(axes.c2p(2.5, compute_roof_y), color=BLUE_D, stroke_width=4)

        compute_label = Text("P_peak = 19.5K GFLOP/s", font_size=24, color=BLUE_D)
        compute_label.next_to(axes.c2p(1.5, compute_roof_y), UP, buff=0.1)
        compute_label.set_stroke(BLACK, width=3, background=True)

        self.play(
            Create(compute_roof),
            Write(compute_label),
        )
        self.wait(1)

        # Step 5: Draw the memory roof (sloped line)
        x_range_roof = np.linspace(0, np.log10(I_critical), 100)
        y_range_roof = np.log10(B_peak) + x_range_roof

        memory_roof_points = [axes.c2p(x, y) for x, y in zip(x_range_roof, y_range_roof)]
        memory_roof = VMobject(color=RED_D, stroke_width=4)
        memory_roof.set_points_as_corners(memory_roof_points)

        memory_equation = Text("P = B_peak × I", font_size=24, color=RED_D)
        memory_equation.next_to(axes.c2p(0.3, 2.5), RIGHT + DOWN * 0.5, buff=0.1)
        memory_equation.set_stroke(BLACK, width=3, background=True)

        bandwidth_label = Text("B_peak = 1555 GB/s", font_size=20, color=RED_D)
        bandwidth_label.next_to(memory_equation, DOWN * 0.5, buff=0.1)
        bandwidth_label.set_stroke(BLACK, width=3, background=True)

        self.play(
            Create(memory_roof),
            Write(memory_equation),
            Write(bandwidth_label)
        )
        self.wait(1)

        # Step 6: Mark the critical intensity
        critical_point = axes.c2p(np.log10(I_critical), compute_roof_y)
        critical_dot = Dot(critical_point, color=YELLOW, radius=0.1)

        critical_label = Text("I* = 12.5 FLOP/B", font_size=24, color=YELLOW)
        critical_label.next_to(critical_point, DOWN + RIGHT, buff=0.6)
        critical_label.set_stroke(BLACK, width=3, background=True)

        # Add background rectangle for better visibility
        critical_bg = BackgroundRectangle(critical_label, color=BLACK, fill_opacity=0.7, buff=0.1)
        critical_group = VGroup(critical_bg, critical_label)

        critical_line = DashedLine(
            axes.c2p(np.log10(I_critical), 0),
            critical_point,
            color=YELLOW,
            stroke_width=2,
            dash_length=0.15
        )

        self.play(
            Create(critical_dot),
            Create(critical_line),
            FadeIn(critical_bg),
            Write(critical_label)
        )
        self.wait(1)

        # Step 7: Add region labels
        memory_bound_text = Text("Memory\nBound", font_size=18, color=RED_D, weight=BOLD)
        memory_bound_text.move_to(axes.c2p(0.6, 1.4))

        compute_bound_text = Text("Compute\nBound", font_size=18, color=BLUE_D, weight=BOLD)
        compute_bound_text.move_to(axes.c2p(1.8, 1.4))

        self.play(
            Write(memory_bound_text),
            Write(compute_bound_text)
        )
        self.wait(1)

        # Clear initial explanation
        self.play(FadeOut(explanation_group))

        # Step 8: Show puzzle implementations
        puzzle_info = VGroup()

        puzzle_title = Text("Our Matrix Multiplication:", font_size=20, color=WHITE, weight=BOLD)
        puzzle_title.to_edge(RIGHT, buff=0.5).shift(UP * 2.5)

        puzzle_info.add(puzzle_title)

        # Naïve implementation
        puzzle_naive_I = 0.1875
        puzzle_naive_I_display = 0.1875
        puzzle_naive_P = B_peak * puzzle_naive_I_display
        puzzle_naive_x = 0.15
        puzzle_naive_point = axes.c2p(puzzle_naive_x, np.log10(puzzle_naive_P))
        puzzle_naive_dot = Dot(puzzle_naive_point, color=ORANGE, radius=0.1)

        naive_details = Text("Naïve: I = 0.1875 FLOP/B", font_size=18, color=ORANGE)
        naive_details.next_to(puzzle_title, DOWN, aligned_edge=LEFT, buff=0.3)

        naive_perf = Text("Performance: ~ 292 GFLOP/s", font_size=16, color=ORANGE)
        naive_perf.next_to(naive_details, DOWN, aligned_edge=LEFT, buff=0.1)

        puzzle_info.add(naive_details, naive_perf)

        self.play(
            Write(puzzle_title),
            Write(naive_details),
            Write(naive_perf),
            Create(puzzle_naive_dot)
        )
        self.wait(1)

        # Shared memory implementation
        puzzle_shared_I_display = 0.375
        puzzle_shared_P = B_peak * puzzle_shared_I_display
        puzzle_shared_x = 0.4
        puzzle_shared_point = axes.c2p(puzzle_shared_x, np.log10(puzzle_shared_P))
        puzzle_shared_dot = Dot(puzzle_shared_point, color=TEAL, radius=0.1)

        shared_details = Text("Shared Memory: I = 0.375 FLOP/B", font_size=18, color=TEAL)
        shared_details.next_to(naive_perf, DOWN, aligned_edge=LEFT, buff=0.2)

        shared_perf = Text("Performance: ~ 583 GFLOP/s", font_size=16, color=TEAL)
        shared_perf.next_to(shared_details, DOWN, aligned_edge=LEFT, buff=0.1)

        puzzle_info.add(shared_details, shared_perf)

        self.play(
            Write(shared_details),
            Write(shared_perf),
            Create(puzzle_shared_dot)
        )

        # Show improvement arrow
        improvement_arrow = Arrow(
            puzzle_naive_point,
            puzzle_shared_point,
            color=YELLOW,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.2
        )

        improvement_text = Text("2x faster!", font_size=16, color=YELLOW, weight=BOLD)
        improvement_text.next_to(improvement_arrow.get_center(), UP, buff=0.2)

        self.play(
            Create(improvement_arrow),
            Write(improvement_text)
        )
        self.wait(3)

        # Step 9: Show optimization strategies
        strategies_title = Text("Optimization Strategies:", font_size=18, color=WHITE, weight=BOLD)
        strategies_title.next_to(shared_perf, DOWN, aligned_edge=LEFT, buff=0.3)

        strategy1 = Text("• Increase Arithmetic Intensity", font_size=16, color=GREEN_D)
        strategy1.next_to(strategies_title, DOWN, aligned_edge=LEFT, buff=0.1)

        strategy2 = Text("• Use Blocking / Tiling", font_size=16, color=GREEN_D)
        strategy2.next_to(strategy1, DOWN, aligned_edge=LEFT, buff=0.05)

        strategy3 = Text("• Data Reuse. Stay Tuned!", font_size=16, color=GREEN_D)
        strategy3.next_to(strategy2, DOWN, aligned_edge=LEFT, buff=0.05)

        puzzle_info.add(strategies_title, strategy1, strategy2, strategy3)

        self.play(
            Write(strategies_title),
            Write(strategy1),
            Write(strategy2),
            Write(strategy3)
        )

        # Final goal arrow
        goal_point = axes.c2p(np.log10(I_critical), compute_roof_y)
        goal_arrow = CurvedArrow(
            puzzle_shared_point,
            goal_point,
            angle=-TAU/6,
            color=GOLD,
            stroke_width=3
        )

        goal_text = Text("Goal: Reach the roof!", font_size=18, color=GOLD, weight=BOLD)
        goal_text.next_to(goal_point, UP + LEFT, buff=0.2)

        self.play(
            Create(goal_arrow),
            Write(goal_text)
        )

        self.wait(60)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )


if __name__ == "__main__":
    with tempconfig({
        "preview": True,
        "format": "gif",
        "media_dir": "./media",
        "quality": "medium_quality",
        "output_file": "roofline_model_viz"
    }):
        scene = RooflineModelVisualization()
        scene.render()
