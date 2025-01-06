from manim import *
import numpy as np

########## INTRO #################


#####
# brachistochrome problem
class BrachistochroneVsStraightLine(Scene):
    def construct(self):
        # Create the brachistochrone curve
        def param_func(t):
            x = 2 * (t - np.sin(t))
            y = -2 * (1 - np.cos(t))
            return np.array([x, y, 0])

        curve = ParametricFunction(param_func, t_range=[0, PI], color=BLUE)

        # Create the straight line
        start_point = curve.get_start()
        end_point = curve.get_end()
        line = Line(start_point, end_point, color=GREEN)

        # Group the paths and center them
        paths = VGroup(curve, line).move_to(ORIGIN)

        # Create the balls
        ball_curve = Dot(color=RED).move_to(curve.get_start())
        ball_line = Dot(color=YELLOW).move_to(line.get_start())

        # Add everything to the scene
        self.play(
            Create(curve),
            Create(line),
            FadeIn(ball_curve),
            FadeIn(ball_line),
        )

        # Custom rate function for the brachistochrone
        def brachistochrone_rate(t):
            return 1 - (1 - t) ** 1.3

        # Custom rate function for the straight line
        def straight_line_rate(t):
            return t**2

        # Animate the balls along their paths simultaneously
        self.play(
            MoveAlongPath(ball_curve, curve, rate_func=brachistochrone_rate),
            MoveAlongPath(ball_line, line, rate_func=straight_line_rate),
            run_time=2,
        )

        self.wait(1)


####titles and parts
class Parts(Scene):
    def construct(self):
        title = Text("Table of Contents", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        part1 = Text("Part I: Infinite Series", font_size=48, color=BLUE)
        part2 = Text("Part II: Complex Variables", font_size=48, color=GREEN)

        part1.move_to(ORIGIN + UP)  # Part I above the origin
        part2.move_to(ORIGIN)  # Part II at the origin

        # Animate the parts appearing one by one
        self.play(Write(part1))
        self.wait(2)

        self.play(Write(part2))
        self.wait(2)

        # Fade out part 2 and part 3, and make part 1 larger at the origin
        self.play(FadeOut(part2), part1.animate.move_to(ORIGIN).scale(1.5))
        self.wait(3)
        self.play(FadeOut(part1), FadeOut(title))


class Parts2(Scene):
    def construct(self):
        title = Text("Table of Contents", font_size=48)
        title.to_edge(UP)
        self.play(FadeIn(title))
        self.wait(1)

        part1 = Text("Part I: Infinite Series", font_size=48, color=BLUE)
        part2 = Text("Part II: Complex Variables", font_size=48, color=GREEN)

        part1.move_to(ORIGIN + UP)  # Part I above the origin
        part2.move_to(ORIGIN)  # Part II at the origin

        self.play(FadeIn(part1), FadeIn(part2))
        self.wait(2)

        # Fade out part 2 and part 3, and make part 1 larger at the origin
        self.play(FadeOut(part1), part2.animate.move_to(ORIGIN).scale(1.5))
        self.wait(3)
        self.play(FadeOut(part2), FadeOut(title))


######
# zeta 2
class Zeta2(Scene):
    def construct(self):
        # Define the sum text
        sum_expression = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2}")
        sum_expression.move_to(ORIGIN)

        # Add the sum text to the scene
        self.play(Write(sum_expression))
        self.wait(1)

        # Move the sum to the left
        self.play(sum_expression.animate.shift(LEFT * 3))
        self.wait(1)

        # Create the approximation symbol and decimal number
        approx_symbol = MathTex(r"\approx")
        decimal = DecimalNumber(
            0.5, show_ellipsis=True, num_decimal_places=6, include_sign=False
        )

        # Position the approximation symbol and decimal next to the sum
        approx_symbol.next_to(sum_expression, RIGHT, buff=1)
        decimal.next_to(approx_symbol, RIGHT, buff=0.2)

        # Group the approximation symbol and decimal
        approx_group = VGroup(approx_symbol, decimal)

        # Add the approximation group to the scene
        self.add(approx_group)

        # Define the updater for the decimal number
        def update_decimal(mob, dt):
            current_value = mob.get_value()
            target_value = 1.644934
            increment = dt * 0.3  # Adjust speed by changing increment factor
            new_value = min(current_value + increment, target_value)
            mob.set_value(new_value)

        # Add the updater to the DecimalNumber
        decimal.add_updater(update_decimal)

        # Let the scene play with the updater active
        self.wait(6)  # Adjust duration to match the increment speed

        # Remove the updater
        decimal.remove_updater(update_decimal)

        # Transform the approximation group to the exact fraction
        exact_form = MathTex(r"= \frac{\pi^2}{6}")
        exact_form.move_to(approx_group)

        self.play(Transform(approx_group, exact_form))
        self.wait(2)


class Notation(Scene):
    def construct(self):
        # Create the MathTex objects
        notation_1 = MathTex(r"f(x)", font_size=72)
        notation_2 = MathTex(r"\sum_{n=1}^{\infty} a_n", font_size=72)
        notation_3 = MathTex(r"e \quad \pi", font_size=72)

        # Initially position the integrals at the origin (center of the screen)
        notation_1.move_to(ORIGIN)
        notation_2.move_to(ORIGIN)
        notation_3.move_to(ORIGIN)

        # animate them moving left and right
        self.play(FadeIn(notation_1))
        self.play(notation_1.animate.shift(LEFT * 4.5))  # Move left
        self.wait(1)

        self.play(FadeIn(notation_2))
        self.play(notation_2.animate.shift(RIGHT * 4.5))  # Move right
        self.wait(1)

        self.play(FadeIn(notation_3))
        self.wait(3)

        # Fade out everything at the end
        self.play(FadeOut(notation_1), FadeOut(notation_2), FadeOut(notation_3))


###euler's identity
class EulerId(Scene):
    def construct(self):
        # Create the MathTex objects
        eulerid = MathTex(r"e^{i \pi}+1 = 0", font_size=72)

        # Initially position the integrals at the origin (center of the screen)
        eulerid.move_to(ORIGIN)

        # animate them moving left and right
        self.play(Write(eulerid))
        self.wait(3)

        # Fade out everything at the end
        self.play(FadeOut(eulerid))


####euler's method


class EulersMethodWithExactSolution(Scene):
    def construct(self):
        # Define the functions
        def f(x, y):
            return x - y

        def exact_solution(x):
            return x - 1 + 2 * np.exp(-x)

        # Initial condition and step size
        x0, y0 = 0, 1
        h = 0.5
        n_steps = 6  # Number of steps (we will compute for x = 0.5, 1, 1.5, 2, 2.5, 3)

        # Create axes
        axes = Axes(
            x_range=[0, 3],
            y_range=[-1, 2],
            axis_config={"color": WHITE},
        )

        # Create the graph for Euler's method approximations
        euler_dots = VGroup()
        line_segments = VGroup()

        # Initial point for Euler's method
        x, y = x0, y0
        last_dot = Dot(color=RED).move_to(axes.c2p(x, y))
        euler_dots.add(last_dot)

        # Plot the exact solution
        exact_curve = axes.plot(
            lambda x: exact_solution(x), color=GREEN, x_range=[0, 3]
        )

        # Create the label for the equation y' = x - y
        label = MathTex("y' = x - y", color=GREEN).to_edge(UP)

        # Animate the creation of axes, label, and exact solution
        self.play(Create(axes), Write(label), Create(exact_curve))

        # Animate the first dot for Euler's method
        self.play(FadeIn(last_dot))

        # Perform Euler's method and animate each step
        for i in range(n_steps):
            # Calculate the next value using Euler's method
            y_next = y + h * f(x, y)
            x_next = x + h

            # Create a new dot for the next point
            dot_new = Dot(color=YELLOW).move_to(axes.c2p(x_next, y_next))
            euler_dots.add(dot_new)

            # Draw the line between the last point and the new point
            line = Line(last_dot.get_center(), dot_new.get_center(), color=BLUE)
            line_segments.add(line)

            # Animate the line and dot movement
            self.play(Create(line), Transform(last_dot, dot_new), run_time=0.5)

            # Update x and y for the next step
            x, y = x_next, y_next

        # Wait at the end
        self.wait(2)


####euler flow diff eq
class ContinuousMotion(Scene):
    def construct(self):
        # Define the function for the vector field
        func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT

        # Create StreamLines with the vector field
        stream_lines = StreamLines(func, stroke_width=3, max_anchors_per_line=30)

        # Add the streamlines to the scene
        self.add(stream_lines)

        # Start the animation for streamlines
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)

        # Wait until the streamlines animation completes
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)

        # Define the equations to be written at the origin
        eq1 = MathTex(
            r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla w + \mathbf{g}",
            font_size=72,
            color=WHITE,
        ).move_to(ORIGIN)

        eq2 = MathTex(
            r"\nabla \cdot \mathbf{u} = 0", font_size=72, color=WHITE
        ).next_to(eq1, DOWN)

        # Write the equations in the origin (center of the screen)
        self.play(Write(eq1))
        self.wait(2)
        self.play(Write(eq2))

        # Wait a bit before finishing the scene
        self.wait(2)


####prime numbers
class HighlightPrimes(Scene):
    def construct(self):
        # Generate a list of numbers 1 through 100
        numbers = list(range(1, 101))

        # Helper function to check if a number is prime
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        # Create a grid of numbers
        grid = VGroup()
        rows, cols = 10, 10  # 10x10 grid
        for i in range(rows):
            for j in range(cols):
                num = numbers[i * cols + j]
                text = Text(str(num), font_size=24)
                # Adjust the positions so that it spans the whole 10x10 grid
                text.move_to(np.array([j - cols / 2, rows / 2 - i, 0]))
                grid.add(text)

        grid.move_to(ORIGIN)
        grid.scale(0.80)

        # Animate writing the grid numbers
        self.play(*[Write(num) for num in grid], run_time=5)
        self.wait(1)

        # Highlight the prime numbers
        for num in grid:
            value = int(num.text)
            if is_prime(value):
                self.play(num.animate.set_color(ORANGE), run_time=0.2)

        self.wait(2)


##### PART 1: INFINITE SERIES ##################
from manim import *


class GeoSeries(Scene):
    def construct(self):
        # Create the MathTex objects
        title = Text("Geometric Series")
        gs1 = MathTex(r"\sum_{n=0}^\infty a r^{n}")
        gs2 = MathTex(r"\sum_{n=1}^\infty a r^{n-1}")

        # Initially
        gs1.move_to(ORIGIN)
        gs2.move_to(ORIGIN)  # Move gs2 to the top-left
        title.to_edge(UP)

        # animate them moving left and right
        self.play(Write(title))
        self.play(FadeIn(gs1))
        self.wait(2)
        self.play(FadeOut(gs1))

        self.play(FadeIn(gs2))
        self.wait(2)

        self.play(gs2.animate.to_edge(UP + LEFT).shift(DOWN * 1.5))

        partial_sum = MathTex(
            r"""
            s_n &= \frac{a(1 - r^n)}{1 - r} \\
            &= \frac{a}{1-r} - \frac{ar^n}{1-r}
            """
        )
        partial_sum.next_to(gs2, RIGHT, buff=2.5)
        self.play(Write(partial_sum))
        self.wait(4)

        # Show the condition for convergence
        condition = MathTex(r"-1 < r < 1")
        condition.next_to(gs2, DOWN, buff=0.5)
        self.play(Write(condition))
        self.wait(2)

        # Transform partial_sum into partial_sum_lim
        partial_sum_lim = MathTex(
            r"\lim_{n\to \infty} s_n = \lim_{n \to \infty} \left(\frac{a}{1-r} - \frac{ar^n}{1-r}\right)"
        )
        partial_sum_lim.next_to(title, DOWN, buff=0.5)
        self.play(Transform(partial_sum, partial_sum_lim))
        self.wait(3)

        # Now fully fade out partial_sum_lim before continuing with the next transition
        self.play(FadeOut(partial_sum_lim))

        ps_lim2 = MathTex(
            r"= \lim_{n \to \infty} \frac{a}{1-r} - \lim_{n \to \infty} \frac{ar^n}{1-r}"
        )
        ps_lim2.next_to(partial_sum_lim, DOWN, buff=0.5)

        # Fade in ps_lim2 after fading out partial_sum_lim
        self.play(FadeIn(ps_lim2))
        self.wait(2)

        ps_lim3 = MathTex(r"= \frac{a}{1-r} - \frac{a}{1-r} \lim_{n \to \infty} r^n")
        ps_lim3.next_to(partial_sum_lim, DOWN, buff=0.5)

        # Now transform ps_lim2 into ps_lim3
        self.play(Transform(ps_lim2, ps_lim3))
        self.wait(2)

        ps_lim4 = MathTex(r"\lim_{n \to \infty}s_n = \frac{a}{1-r}")
        ps_lim4.next_to(ps_lim3, DOWN, buff=0.5)
        self.play(FadeIn(ps_lim4))
        self.wait(4)

        final = MathTex(
            r"\sum_{n=1}^\infty ar^{n-1} = \sum_{n=0}^\infty ar^n = \frac{a}{1-r}"
        )
        final.next_to(ps_lim3, DOWN, buff=0.5)

        self.play(Transform(ps_lim4, final))
        self.wait(5)


class BernoulliSum(Scene):
    def construct(self):
        # Create the MathTex objects
        sum_1 = MathTex(r"\sum_{n=1}^\infty \frac{n^2}{2^n} = 6", font_size=72)
        sum_2 = MathTex(r"\sum_{n=1}^{\infty} \frac{n^3}{2^n} = 26", font_size=72)

        # Initially position the integrals at the origin (center of the screen)
        sum_1.move_to(ORIGIN)
        sum_2.move_to(ORIGIN)

        # animate them moving left and right
        self.play(FadeIn(sum_1))
        self.play(sum_1.animate.shift(LEFT * 4))  # Move left
        self.wait(1)

        self.play(FadeIn(sum_2))
        self.play(sum_2.animate.shift(RIGHT * 4))  # Move right
        self.wait(3)

        # Fade out everything at the end
        self.play(FadeOut(sum_1), FadeOut(sum_2))


class BaselIntro(Scene):
    def construct(self):
        # Create the MathTex objects
        summation = MathTex(
            r"1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \ldots + \frac{1}{n^2}+ \ldots = ?",
            font_size=72,
        )

        # Initially position the integrals at the origin (center of the screen)
        summation.move_to(ORIGIN)

        # animate them moving left and right
        self.play(Write(summation))
        self.wait(2)

        # Fade out everything at the end
        self.play(FadeOut(summation))

        title = MathTex(
            r"\text{Bounding the Monster}: \sum_{n=1}^\infty \frac{1}{n^2}",
            font_size=48,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        equality = MathTex(r"2n^2 \geq n(n+1)")
        equality.next_to(title, DOWN, buff=0.75)

        self.play(Write(equality))
        self.wait(3)

        equality2 = MathTex(r"\frac{1}{n^2} \leq \frac{1}{n(n+1)/2}")
        equality2.next_to(equality, DOWN, buff=0.5)
        self.play(Write(equality2))
        self.wait(2)
        self.play(FadeOut(equality), FadeOut(equality2))

        # First part of the sum before \leq
        sum1_left = MathTex(
            r"1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \ldots + \frac{1}{n^2}+\ldots"
        )

        # Second part of the sum after \leq
        sum1_right = MathTex(
            r"\leq 1 + \frac{1}{3} + \frac{1}{6} + \ldots + \frac{1}{n(n+1)/2} + \ldots"
        )

        # Position sum1_left and sum1_right
        sum1_left.next_to(title, DOWN, buff=0.75)
        sum1_right.next_to(sum1_left, DOWN, buff=0.5)

        # Write the first part (before \leq)
        self.play(Write(sum1_left))
        self.wait(2)

        # Then write the second part (after \leq)
        self.play(Write(sum1_right))
        self.wait(4)

        sum1_right_transformed = MathTex(r"\leq 2")
        sum1_right_transformed.next_to(sum1_left, DOWN, buff=0.75)

        # Transform the right part into the new expression
        self.play(Transform(sum1_right, sum1_right_transformed))
        self.wait(2)

        self.play(FadeOut(sum1_left))

        bound = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2}")
        bound.next_to(sum1_right_transformed, LEFT, buff=0.3)
        self.play(Write(bound))
        self.wait(5)

        converges = MathTex(
            r"\text{Similarly Converges for} \sum_{n=1}^\infty \frac{1}{n^p}, \text{ for } p = 3,4,5,\ldots"
        )
        converges.next_to(title, DOWN, buff=0.5)

        self.play(Write(converges))
        self.wait(3)


class BaselMystery(Scene):
    def construct(self):
        # Create the MathTex objects
        text = Text(
            "If anyone finds and commnicates to us \nthat which thus far  has eluded our efforts, \ngreat will be our gratitude.\n-Jakob Bernoulli (1689)"
        )

        # Initially position the integrals at the origin (center of the screen)
        text.move_to(ORIGIN)

        # animate them moving left and right
        self.play(Write(text))
        self.wait(5)

        self.play(FadeOut(text))


class EulerApproximation(Scene):
    def construct(self):
        title = MathTex(
            r"\text{Euler's Approximation of } \sum_{n=1}^\infty \frac{1}{n^2}",
            font_size=48,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # sum up to 1
        sum_expression1 = MathTex(r"\sum_{n=1}^{1} \frac{1}{n^2}")
        sum_expression1.next_to(title, DOWN + LEFT, buff=0.5)

        # Add the sum text to the scene
        self.play(Write(sum_expression1))
        self.wait(1)

        # Create the approximation symbol and decimal number
        approx_symbol = MathTex(r"=")
        decimal = DecimalNumber(
            0.8, show_ellipsis=True, num_decimal_places=6, include_sign=False
        )

        # Position the approximation symbol and decimal next to the sum
        approx_symbol.next_to(sum_expression1, RIGHT, buff=1)
        decimal.next_to(approx_symbol, RIGHT, buff=0.2)

        # Group the approximation symbol and decimal
        approx_group = VGroup(approx_symbol, decimal)

        # Add the approximation group to the scene
        self.add(approx_group)

        # Define the updater for the decimal number
        def update_decimal(mob, dt):
            current_value = mob.get_value()
            target_value = 1
            increment = dt * 0.3  # Adjust speed by changing increment factor
            new_value = min(current_value + increment, target_value)
            mob.set_value(new_value)

        # Add the updater to the DecimalNumber
        decimal.add_updater(update_decimal)

        # Let the scene play with the updater active
        self.wait(3)  # Adjust duration to match the increment speed

        # Remove the updater
        decimal.remove_updater(update_decimal)

        # sum up to 10

        sum_expression2 = MathTex(r"\sum_{n=1}^{10} \frac{1}{n^2}")
        sum_expression2.next_to(sum_expression1, DOWN, buff=0.3)
        self.play(Write(sum_expression2))
        self.wait(1)
        approx_symbol = MathTex(r"\approx")
        decimal = DecimalNumber(
            0.5, show_ellipsis=True, num_decimal_places=15, include_sign=False
        )

        # Position the approximation symbol and decimal next to the sum
        approx_symbol.next_to(sum_expression2, RIGHT, buff=1)
        decimal.next_to(approx_symbol, RIGHT, buff=0.2)

        approx_group = VGroup(approx_symbol, decimal)
        # Add the approximation group to the scene
        self.add(approx_group)

        # Define the updater for the decimal number
        def update_decimal(mob, dt):
            current_value = mob.get_value()
            target_value = 1.5497677311665406903502141597379692
            increment = dt * 0.3  # Adjust speed by changing increment factor
            new_value = min(current_value + increment, target_value)
            mob.set_value(new_value)

        # Add the updater to the DecimalNumber
        decimal.add_updater(update_decimal)

        # Let the scene play with the updater active
        self.wait(6)  # Adjust duration to match the increment speed

        # Remove the updater
        decimal.remove_updater(update_decimal)

        # sum up to 100

        sum_expression3 = MathTex(r"\sum_{n=1}^{100} \frac{1}{n^2}")
        sum_expression3.next_to(sum_expression2, DOWN, buff=0.5)
        self.play(Write(sum_expression3))
        self.wait(1)
        approx_symbol = MathTex(r"\approx")
        decimal = DecimalNumber(
            0.5, show_ellipsis=True, num_decimal_places=20, include_sign=False
        )

        # Position the approximation symbol and decimal next to the sum
        approx_symbol.next_to(sum_expression3, RIGHT, buff=1)
        decimal.next_to(approx_symbol, RIGHT, buff=0.2)

        approx_group = VGroup(approx_symbol, decimal)
        # Add the approximation group to the scene
        self.add(approx_group)

        # Define the updater for the decimal number
        def update_decimal(mob, dt):
            current_value = mob.get_value()
            target_value = 1.6349839001848928650771694981803
            increment = dt * 0.3  # Adjust speed by changing increment factor
            new_value = min(current_value + increment, target_value)
            mob.set_value(new_value)

        # Add the updater to the DecimalNumber
        decimal.add_updater(update_decimal)

        # Let the scene play with the updater active
        self.wait(10)  # Adjust duration to match the increment speed

        # Remove the updater
        decimal.remove_updater(update_decimal)
        self.wait(5)


class Baselapprxbad(Scene):
    def construct(self):
        # Create the MathTex objects
        text = Text(
            "Clearly, numerical approximations \nwon't give us the exact answer."
        )

        text.move_to(ORIGIN)

        # animate them moving left and right
        self.play(Indicate(text))
        self.wait(5)

        self.play(FadeOut(text))

        text2 = Text("So, is there a better way besides brute forcing?")
        text2.move_to(ORIGIN)

        self.play(Write(text2))

        self.wait(5)
        self.play(FadeOut(text2))

        text3 = Text(
            "Yes! This was how a 24 year old \nEuler handled this approximation."
        )
        text3.move_to(ORIGIN)

        self.play(Write(text3))

        self.wait(5)
        self.play(FadeOut(text3))


class BaselApprxI(Scene):
    def construct(self):
        title = Text(
            "Step 1: Improper Integral and Series Trick",
            font_size=48,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        expression1 = MathTex(r"I = \int_{0}^{1/2} - \frac{\ln(1-t)}{t} \, dt")
        expression1.next_to(title, DOWN, buff=0.5)
        self.play(Write(expression1))
        self.wait(1)

        expression2 = MathTex(
            r"= \int_{0}^{1/2} - \frac{-t - \frac{t^2}{2} - \frac{t^3}{3} - \frac{t^4}{4} - \ldots}{t} \, dt"
        )
        expression2.next_to(expression1, DOWN, buff=0.5)
        self.play(Write(expression2))
        self.wait(3)

        expression3 = MathTex(
            r"= \int_{0}^{1/2} 1 + \frac{t}{2} + \frac{t^2}{3} + \frac{t^3}{4} + \ldots \, dt"
        )
        expression3.next_to(expression1, DOWN, buff=0.5)
        self.play(Transform(expression2, expression3))
        self.wait(3)

        expression4 = MathTex(
            r"= t + \frac{t^2}{4} + \frac{t^3}{9} + \frac{t^4}{16} + \ldots \Big|_{0}^{1/2}"
        )
        expression4.next_to(expression3, DOWN, buff=0.5)
        self.play(Write(expression4))
        self.wait(5)

        expression5 = MathTex(
            r"= \left(\frac{1}{2}\right) + \frac{\left(\frac{1}{2}\right)^2}{4} + \frac{\left(\frac{1}{2}\right)^3}{9} + \frac{\left(\frac{1}{2}\right)^4}{16} + \ldots"
        )
        expression5.next_to(expression3, DOWN, buff=0.5)
        self.play(Transform(expression4, expression5))
        self.wait(3)


class BaselApprxII(Scene):
    def construct(self):
        title = Text(
            "Step 2: Substitution",
            font_size=48,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        expression0 = MathTex(r"\text{Let } z = 1 - t")
        expression0.next_to(title, DOWN, buff=0.5)
        self.play(Write(expression0))
        self.wait(1)
        self.play(expression0.animate.to_edge(UP + LEFT).shift(DOWN * 1.5))

        # Second expression: integral expression
        expression1 = MathTex(r"I = \int_{0}^{1/2} - \frac{\ln(1-t)}{t} \, dt")
        expression1.next_to(title, DOWN, buff=0.5)
        self.play(Write(expression1))
        self.wait(2)

        # Third expression: transformation of expression1 to expression2
        expression2 = MathTex(r"I = \int_{1}^{1/2} \frac{\ln(z)}{1-z} \, dz")
        expression2.next_to(title, DOWN, buff=0.5)
        self.play(Transform(expression1, expression2))
        self.play(FadeOut(expression2))
        self.wait(3)

        # Fourth expression: transformation to expression3
        expression3 = MathTex(r" = \int_{1}^{1/2} \frac{1}{1-z} \ln(z) \, dz")
        expression3.next_to(expression2, RIGHT, buff=0.3)
        self.play(Write(expression3))
        self.wait(2)

        # Fifth expression: expanding the series inside the integral
        expression4 = MathTex(r"= \int_{1}^{1/2} (1+z+z^2+z^3+\ldots)\ln(z) \, dz")
        expression4.next_to(expression2, DOWN, buff=0.5)
        self.play(Write(expression4))
        self.wait(2)

        # Sixth expression: expanding the sum into separate integrals
        expression5 = MathTex(
            r"= \int_{1}^{1/2} \ln(z) \, dz + \int_{1}^{1/2} z\ln(z) \, dz + \int_{1}^{1/2} z^2\ln(z) \, dz + \ldots"
        )
        expression5.next_to(expression4, DOWN, buff=0.5)
        self.play(Write(expression5))
        self.wait(2)

        # Seventh expression: showing the integration by parts formula (IBP)
        expression6 = MathTex(
            r"\text{Note IBP: } \int_{1}^{1/2} z^n \ln(z) \, dz = \frac{z^{n+1}}{n+1} \ln(z) - \frac{z^{n+1}}{(n+1)^2} \Big|_{1}^{1/2}"
        )
        expression6.next_to(expression5, DOWN, buff=0.3)
        self.play(Write(expression6))
        self.wait(8)

        self.play(
            FadeOut(expression0),
            FadeOut(expression1),
            FadeOut(expression2),
            FadeOut(expression3),
            FadeOut(expression4),
            FadeOut(expression5),
            FadeOut(expression6),
        )
        expression7 = MathTex(
            r"I = (z\ln(z) -z) + \left(\frac{z^2}{2} \ln(z) - \frac{z^2}{4}\right) + \left(\frac{z^3}{3} \ln(z) - \frac{z^3}{9}\right) + \ldots \Big|_{1}^{1/2}",
            font_size=36,
        )
        expression7.next_to(title, DOWN, buff=0.5)
        self.play(Write(expression7))
        self.wait(4)

        expression8 = MathTex(
            r"= \ln(z) \left[z + \frac{z^2}{2} + \frac{z^3}{3} + \ldots\right] - \left[z+ \frac{z^2}{4} +\frac{z^3}{9} + \ldots\right]\Big|_{1}^{1/2}"
        )
        expression8.next_to(expression7, DOWN, buff=0.5)
        self.play(Write(expression8))
        self.wait(4)

        expression9 = MathTex(
            r"= \ln(z) \left[- \ln(1-z)\right] - \left[z+ \frac{z^2}{4} +\frac{z^3}{9} + \ldots\right]\Big|_{1}^{1/2}"
        )
        expression9.next_to(expression7, DOWN, buff=0.5)
        self.play(Transform(expression8, expression9))
        self.wait(4)

        # Write expression10
        expression10 = MathTex(
            r"= -\left[\ln \left(\frac{1}{2}\right)\right]^2 - \left[\left(\frac{1}{2}\right) + \frac{\left(\frac{1}{2}\right)^2}{4} +\frac{\left(\frac{1}{2}\right)^3}{9} + \ldots \right] \\+[\ln(1)][\ln(0)] + \sum_{n=1}^\infty \frac{1}{n^2}"
        )
        expression10.next_to(expression9, DOWN, buff=0.5)
        self.play(Write(expression10))
        self.wait(1)

        # Fade out expression7, expression8, and expression9
        self.play(FadeOut(expression7), FadeOut(expression8), FadeOut(expression9))

        # Move expression10 below the title
        self.play(expression10.animate.move_to(title.get_center() + DOWN * 2))
        self.wait(4)

        note = MathTex(
            r"\lim_{z \to 1^-}[\ln(z)][\ln(1-z)] = 0 \implies '[\ln(1)][\ln(0)]= 0'"
        )
        note.next_to(expression10, DOWN, buff=0.5)

        self.play(FadeIn(note))
        self.wait(7)
        self.play(FadeOut(note))
        self.wait(2)

        expression11 = MathTex(
            r"= -\left[\ln \left(\frac{1}{2}\right)\right]^2 - \left[\left(\frac{1}{2}\right) + \frac{\left(\frac{1}{2}\right)^2}{4} +\frac{\left(\frac{1}{2}\right)^3}{9} + \ldots \right] + \sum_{n=1}^\infty \frac{1}{n^2}",
            font_size=48,
        )
        expression11.next_to(title, DOWN, buff=0.5)
        self.play(Transform(expression10, expression11))
        self.wait(5)


class BaselApprxIII(Scene):
    def construct(self):
        title = Text(
            "Step 3: Result",
            font_size=48,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        expression11 = MathTex(
            r"= I = -\left[\ln \left(\frac{1}{2}\right)\right]^2 - \left[\left(\frac{1}{2}\right) + \frac{\left(\frac{1}{2}\right)^2}{4} +\frac{\left(\frac{1}{2}\right)^3}{9} + \ldots \right] + \sum_{n=1}^\infty \frac{1}{n^2}",
            font_size=48,
        )
        expression11.next_to(title, DOWN, buff=0.5)
        self.play(Write(expression11))
        self.wait(5)

        expression12 = MathTex(
            r"= I = -[\ln(2)]^2 - \left[\left(\frac{1}{2}\right) + \frac{\left(\frac{1}{2}\right)^2}{4} +\frac{\left(\frac{1}{2}\right)^3}{9} + \ldots \right] + \sum_{n=1}^\infty \frac{1}{n^2}",
            font_size=48,
        )
        expression12.next_to(title, DOWN, buff=0.5)
        self.play(Transform(expression11, expression12))
        self.wait(5)

        expression13 = MathTex(
            r"= \sum_{n=1}^{\infty}\frac{1}{n^2}= 2 \left[\frac{\left(\frac{1}{2}\right)^2}{4} +\frac{\left(\frac{1}{2}\right)^3}{9} + \ldots \right] + [\ln(2)]^2",
            font_size=48,
        )
        expression13.next_to(expression12, DOWN, buff=0.5)
        self.play(Write(expression13))
        self.wait(5)

        expression14 = MathTex(
            r"= \sum_{n=1}^{\infty} \frac{1}{n^2 2^{n-1}} + [\ln(2)]^2",
            font_size=48,
        )
        expression14.next_to(expression13, DOWN, buff=0.5)
        self.play(Write(expression14))
        self.wait(7)


class BaselApprxIV(Scene):
    def construct(self):
        title = Text(
            "A Better Approximation",
            font_size=36,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        expression14 = MathTex(
            r"\sum_{n=1}^{\infty} \frac{1}{n^2} ",
            font_size=48,
        )
        expression14.to_edge(LEFT + UP * 1.5, buff=1.5)
        self.play(Write(expression14))
        self.wait(2)

        expression15 = MathTex(
            r"\sum_{n=1}^{\infty} \frac{1}{n^2 2^{n-1}} + [\ln(2)]^2 ",
            font_size=48,
        )
        expression15.to_edge(LEFT + UP * 3.5, buff=1.5)
        self.play(Write(expression15))
        self.wait(2)

        self.play(FadeOut(expression14), FadeOut(expression15))
        grid = NumberPlane(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 0.1],  # Adjusted for derivative values
            axis_config={"include_numbers": False},
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
        )
        self.play(Write(grid))
        self.wait(1)

        # Plot the first function in green
        graph1 = grid.plot(lambda x: 1 / x**2, color=GREEN, x_range=[0.1, 10])
        graph1_label = grid.get_graph_label(
            graph1, label=MathTex(r"f = \frac{1}{x^2}"), x_val=1, direction=LEFT
        )
        self.play(Create(graph1), Write(graph1_label))
        self.wait(2)

        # Plot the second function in orange
        graph2 = grid.plot(
            lambda x: 1 / (x**2 * 2 ** (x - 1)) + (np.log(2)) ** 2,
            color=ORANGE,
            x_range=[0.1, 10],
        )
        graph2_label = grid.get_graph_label(
            graph2,
            label=MathTex(r"g = \frac{1}{x^2 2^{x-1}} + (\ln 2)^2"),
            x_val=2,
            direction=RIGHT + UP * 2,
        )
        self.play(Create(graph2), Write(graph2_label))
        self.wait(3)

        # Fade out the original graphs
        self.play(
            FadeOut(graph1),
            FadeOut(graph2),
            FadeOut(graph1_label),
            FadeOut(graph2_label),
        )
        self.wait(1)
        """
        # Plot the derivative of the first function in green
        derivative1 = grid.plot(
            lambda x: -2 / x**3, color=GREEN, x_range=[0.1, 10]  # Correct derivative
        )
        derivative1_label = grid.get_graph_label(
            derivative1, label=MathTex(r"f'"), x_val=1, direction=LEFT
        )
        self.play(Create(derivative1), Write(derivative1_label))
        self.wait(2)

        # Plot the derivative of the second function
        derivative2 = grid.plot(
            lambda x: -((2 ** (-x + 1)) * (np.log(2) * x + 2)) / x**3,
            color=ORANGE,
            x_range=[0.1, 10],
        )
        derivative2_label = grid.get_graph_label(
            derivative2,
            label=MathTex(r"g'"),
            x_val=2,
            direction=RIGHT * 2,
        )
        self.play(Create(derivative2), Write(derivative2_label))
        self.wait(3)
        """


class FollowingGraphCamera(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        # Create the axes for plotting the derivatives
        ax = Axes(
            x_range=[0, 10],
            y_range=[-2, 2],
            axis_config={"include_numbers": False},
        )
        # Plot derivative1 (green) and derivative2 (orange)
        derivative1 = ax.plot(lambda x: -2 / x**3, color=GREEN, x_range=[0.1, 10])
        derivative2 = ax.plot(
            lambda x: -((2 ** (-x + 1)) * (np.log(2) * x + 2)) / x**3,
            color=ORANGE,
            x_range=[0.1, 10],
        )

        # Create graph labels for derivative1 and derivative2
        derivative1_label = ax.get_graph_label(
            derivative1, label=MathTex(r"f'"), x_val=1, direction=LEFT
        )
        derivative2_label = ax.get_graph_label(
            derivative2, label=MathTex(r"g'"), x_val=2, direction=RIGHT * 2 + UP * 2
        )

        self.play(
            Write(ax),
            Create(derivative1),
            Write(derivative1_label),
            Create(derivative2),
            Write(derivative2_label),
        )
        self.wait(2)

        # Create a moving dot to follow derivative1
        moving_dot = Dot(ax.i2gp(1.5, derivative1), color=WHITE)
        moving_dot.set_opacity(0)
        self.add(moving_dot)

        # Zoom into the initial position of the moving dot on derivative1
        self.play(self.camera.frame.animate.scale(0.5).move_to(moving_dot))

        self.wait(5)

        # Zoom out after the movement
        self.play(Restore(self.camera.frame))
        self.wait()


class BulletPointsScene(Scene):
    def construct(self):
        # Create the bullet points as text objects
        bullet_point_1 = Text(
            "Used Integrals, Logarithms, Series, \nand Integration by Parts",
            font_size=36,
        )
        bullet_point_2 = Text(
            "A 'faster' way for performing this \nnumerical approximation", font_size=36
        )
        bullet_point_3 = Text(
            "Still an estimate :( \nwe need an exact sum !!", font_size=48
        )

        # Position them
        bullet_point_1.to_edge(UP)
        bullet_point_2.next_to(bullet_point_1, DOWN, buff=1)
        bullet_point_3.next_to(bullet_point_2, DOWN, buff=1)

        # Group the bullet points together (optional, for better control)
        bullet_points_group = VGroup(bullet_point_1, bullet_point_2, bullet_point_3)

        # Display the first bullet point
        self.play(Write(bullet_point_1))
        self.wait(3)

        # Display the second bullet point, with a small delay
        self.play(Write(bullet_point_2))
        self.wait(5)

        # Display the third bullet point, with a small delay
        self.play(Write(bullet_point_3))
        self.wait(5)

        # Optionally: animate the bullet points together (e.g., fade them out)
        self.play(FadeOut(bullet_points_group))
        self.wait()

        statement = MathTex(r"\text{How Euler did it:} \sum_{n=1}^\infty \frac{1}{n^2}")
        statement.move_to(ORIGIN)
        self.play(Write(statement))
        self.wait(5)


class BaselSolution(Scene):
    def construct(self):
        title = MathTex(
            r"\text{The Basel Problem: } \sum_{n=1}^\infty \frac{1}{n^2} ",
            font_size=48,
        )
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Expression 1 (to appear below title)
        expression1 = MathTex(
            r"p(x) = 1 - \frac{x^2}{3!} + \frac{x^4}{5!} - \frac{x^6}{7!} + \frac{x^8}{9!} - \ldots"
        )
        expression1.next_to(title, DOWN, buff=0.5)
        self.play(Write(expression1))
        self.wait(2)

        # Expression 2 (Note about p(0))
        expression2 = MathTex(r"\text{Note: } p(0) =1")
        expression2.next_to(expression1, DOWN, buff=0.5)
        self.play(FadeIn(expression2))
        self.wait(2)
        self.play(FadeOut(expression2))  # Fade out expression2
        self.wait(4)

        # Expression 3 (modified p(x) form)
        expression3 = MathTex(
            r"p(x) = x \left[\frac{1 - \frac{x^2}{3!} + \frac{x^4}{5!} - \frac{x^6}{7!} + \frac{x^8}{9!} - \ldots}{x} \right]"
        )
        expression3.next_to(expression1, DOWN, buff=0.5)
        self.play(Write(expression3))
        self.wait(2)

        # Expression 4 (transformation of expression3 to the final form)
        expression4 = MathTex(
            r"p(x) = \frac{x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \frac{x^9}{9!} - \ldots}{x} "
        )
        expression4.next_to(expression1, DOWN, buff=0.5)
        self.play(Transform(expression3, expression4))
        self.wait(3)

        expression5 = MathTex(r"p(x) = \frac{\sin(x)}{x}")
        expression5.next_to(expression4, DOWN, buff=0.5)
        self.play(Write(expression5))
        self.wait(5)


class SincPlotScene(Scene):
    def construct(self):
        # Create the axes/grid for the plot
        axes = Axes(
            x_range=[-10, 10],
            y_range=[-0.5, 1.5],
            axis_config={"include_numbers": False},
        )

        # Plot the sinc function sin(x)/x
        graph = axes.plot(
            lambda x: np.sinc(x / np.pi), color=BLUE
        )  # np.sinc(x) is sin(pi*x)/(pi*x)

        # Add a label to the graph
        graph_label = axes.get_graph_label(
            graph, label=r"\frac{\sin(x)}{x}", x_val=2, direction=UP + RIGHT * 2
        )

        # Play the animation: Write the axes, plot the graph, and display the label
        self.play(Create(axes))
        self.wait(2)

        self.play(Create(graph), Write(graph_label))
        self.wait(5)
        # Fade out everything (axes, graph, label)
        self.play(FadeOut(axes), FadeOut(graph), FadeOut(graph_label))
        self.wait(1)

        text = MathTex(
            r"\text{So, } p(x) = 0 \implies \sin(x) = 0 \\ \implies x = \pm k\pi, \quad k \in \mathbb{N} \\ \text{Note: } x=0 \text{ is NOT a solution, as } p(0)=1"
        )
        text.move_to(ORIGIN)
        self.play(FadeIn(text))
        self.wait(7)


class BaselSolution2(Scene):
    def construct(self):

        expression1 = MathTex(
            r"p(x) = 1 - \frac{x^2}{3!} + \frac{x^4}{5!} - \frac{x^6}{7!} + \frac{x^8}{9!} - \ldots"
        )
        expression1.to_edge(UP)
        self.play(Write(expression1))
        self.wait(2)

        expression2 = MathTex(
            r"=\left(1 - \frac{x}{\pi}\right)\left(1 - \frac{x}{-\pi}\right)\left(1 - \frac{x}{2\pi}\right)\left(1 - \frac{x}{-2\pi}\right)\left(1 - \frac{x}{3\pi}\right)\left(1 - \frac{x}{-3\pi}\right)\ldots",
            font_size=36,
        )
        expression2.next_to(expression1, DOWN, buff=0.5)
        self.play(Write(expression2))
        self.wait(2)

        expression3 = MathTex(
            r"=\left(1 - \frac{x^2}{\pi^2}\right)\left(1 - \frac{x^2}{4\pi^2}\right)\left(1 - \frac{x^2}{9\pi^2}\right)\ldots"
        )
        expression3.next_to(expression1, DOWN, buff=0.5)
        self.play(Transform(expression2, expression3))
        self.wait(4)

        expression4 = MathTex(
            r"= 1 - \left( \frac{1}{\pi^2} + \frac{1}{4\pi^2} + \frac{1}{9\pi^2} + \ldots \right) x^2 + \ldots"
        )
        expression4.next_to(expression3, DOWN, buff=0.5)
        self.play(Write(expression4))
        self.wait(5)

        expression5 = MathTex(
            r"1- \frac{x^2}{3!} = 1 - \left( \frac{1}{\pi^2} + \frac{1}{4\pi^2} + \frac{1}{9\pi^2} + \ldots \right) x^2 + \ldots"
        )
        expression5.next_to(expression4, DOWN, buff=0.5)
        self.play(Write(expression5))
        self.wait(5)


class BaselSolution3(Scene):
    def construct(self):

        expression1 = MathTex(
            r"1- \frac{x^2}{3!} = 1 - \left( \frac{1}{\pi^2} + \frac{1}{4\pi^2} + \frac{1}{9\pi^2} + \ldots \right) x^2 + \ldots"
        )
        expression1.to_edge(UP)
        self.play(Write(expression1))
        self.wait(4)

        expression2 = MathTex(
            r"\frac{1}{3!} = \left( \frac{1}{\pi^2} + \frac{1}{4\pi^2} + \frac{1}{9\pi^2} + \ldots \right) + \ldots"
        )
        expression2.next_to(expression1, DOWN, buff=0.5)
        self.play(Write(expression2))
        self.wait(4)

        expression3 = MathTex(
            r" \frac{1}{3!} = \frac{1}{\pi^2} \left(1 + \frac{1}{4} + \frac{1}{9} + \ldots\right)"
        )
        expression3.next_to(expression1, DOWN, buff=0.5)
        self.play(Transform(expression2, expression3))
        self.wait(3)

        expression4 = MathTex(
            r" \frac{1}{3!} = \frac{1}{\pi^2} \sum_{n=1}^\infty \frac{1}{n^2}"
        )
        expression4.next_to(expression3, DOWN, buff=0.5)
        self.play(Write(expression4))
        self.wait(3)

        expression5 = MathTex(r" \frac{\pi^2}{3!} = \sum_{n=1}^\infty \frac{1}{n^2}")
        expression5.next_to(expression4, DOWN, buff=0.5)
        self.play(Write(expression5))
        self.wait(3)

        expression6 = MathTex(r" \frac{\pi^2}{6} = \sum_{n=1}^\infty \frac{1}{n^2}")
        expression6.next_to(expression4, DOWN, buff=0.5)
        self.play(Transform(expression5, expression6))
        self.wait(5)


class BaselSolutionNEXT(Scene):
    def construct(self):

        expression1 = MathTex(
            r"\zeta (s) = \sum_{n=1}^\infty \frac{1}{n^s} = \frac{1}{\Gamma (s)} \int_{0}^{\infty} \frac{x^{s-1}}{e^x -1}dx \\ \text{where } \Gamma(s) = \int_{0}^{\infty} x^{s-1}e^{-x}dx"
        )
        expression1.move_to(ORIGIN)
        self.play(Write(expression1))
        self.wait(6)

        self.play(FadeOut(expression1))

        expression2 = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^3} = ?")
        expression2.move_to(ORIGIN)
        self.play(Write(expression2))
        self.wait(6)


class BaselSolutionNEXT2(Scene):
    def construct(self):

        expression1 = MathTex(
            r"\sum_{n=1}^\infty \frac{1}{n^{26}} = \frac{1315862}{11094481976030578125}\pi^{26}"
        )
        expression1.move_to(ORIGIN)
        self.play(Write(expression1))
        self.wait(6)

        self.play(FadeOut(expression1))


########Complex Numbers
class ComplexNumIntro(Scene):
    def construct(self):
        # Create and display the expression
        expression1 = MathTex(r"x^2 + 1 = 0")
        expression1.move_to(ORIGIN)
        self.play(Write(expression1))
        self.wait(2)

        # Animate the expression moving to the top edge
        self.play(expression1.animate.to_edge(UP))
        self.wait(1)

        # Create axes for plotting
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3, 2],
            axis_config={"include_numbers": False, "include_tip": False},
        )

        # Define and plot the graph of x^2 + 1
        graph = ax.plot(lambda x: x**2 + 1, color=BLUE, x_range=[-3, 3])

        # Display the axes, graph, and label
        self.play(Create(ax))
        self.wait(1)
        self.play(Create(graph))
        self.wait(5)

        # Fade everything out at the end
        self.play(FadeOut(ax), FadeOut(graph), FadeOut(expression1))
        self.wait()


class ComplexSillyness(Scene):
    def construct(self):
        # Create the MathTex objects
        notation_1 = MathTex(r"e^x = -1", font_size=72)
        notation_2 = MathTex(r"\cos (x) = 2", font_size=72)

        # Initially position the integrals at the origin (center of the screen)
        notation_1.move_to(ORIGIN)
        notation_2.move_to(ORIGIN)

        # animate them moving left and right
        self.play(FadeIn(notation_1))
        self.play(notation_1.animate.shift(LEFT * 4))  # Move left
        self.wait(2)

        self.play(FadeIn(notation_2))
        self.play(notation_2.animate.shift(RIGHT * 4))  # Move right
        self.wait(5)

        # Fade out everything at the end
        self.play(FadeOut(notation_1), FadeOut(notation_2))

        notation_3 = MathTex(r"i^2 = -1", font_size=72)
        notation_3.move_to(ORIGIN)
        self.play(Write(notation_3))
        self.wait(5)
        self.play(FadeOut(notation_3))


class EulerIdentity(Scene):
    def construct(self):
        # Create the complex plane
        plane = ComplexPlane(x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1])
        plane.add_coordinates()
        self.add(plane)

        # Define the circle path (unit circle)
        circle = Circle(radius=1, color=BLUE)
        self.play(Create(circle))

        # Dot to trace the circle
        moving_dot = Dot(color=YELLOW)
        self.add(moving_dot)

        # Trace the path of the dot
        path = TracedPath(moving_dot.get_center, stroke_color=YELLOW, stroke_width=2)
        self.add(path)

        # Real and imaginary projections
        real_projection = always_redraw(
            lambda: DashedLine(
                start=moving_dot.get_center(),
                end=np.array([moving_dot.get_center()[0], 0, 0]),
                color=GREEN,
            )
        )
        imag_projection = always_redraw(
            lambda: DashedLine(
                start=moving_dot.get_center(),
                end=np.array([0, moving_dot.get_center()[1], 0]),
                color=RED,
            )
        )
        self.add(real_projection, imag_projection)

        # Add labels
        real_label = MathTex(r"\cos(x)", color=GREEN).next_to(plane.n2p(1), DOWN)
        imag_label = MathTex(r"\sin(x)", color=RED).next_to(plane.n2p(1j), LEFT)
        self.play(Write(real_label), Write(imag_label))

        # Animate the dot along the unit circle
        self.play(MoveAlongPath(moving_dot, circle, rate_func=linear, run_time=5))

        # Euler's Identity formula
        euler_formula = MathTex(r"e^{ix} = \cos(x) + i\sin(x)", font_size=72).to_edge(
            UP
        )
        self.play(Write(euler_formula))

        self.wait(3)


class DeMoivre(Scene):
    def construct(self):
        # Create and display the title
        title = Text("De Moivre's Formula", font_size=48)
        self.play(Write(title))
        self.wait(2)

        # Animate the title moving to the top edge
        self.play(title.animate.to_edge(UP))
        self.wait(1)

        # Display the formula
        formula = MathTex(
            r"(\cos(\theta) \pm i \sin(\theta))^n = \cos(n\theta) \pm i \sin(n\theta)",
            font_size=48,
        )
        self.play(Write(formula))
        self.wait(5)

        # Add a hint below the formula
        hint = Text(
            "Proof Hint: Use the Pythagorean identity",
            font_size=36,
            color=YELLOW,
        )
        hint.next_to(formula, DOWN, buff=0.5)
        self.play(Write(hint))
        self.wait(3)

        # Fade out the hint and the formula
        self.play(FadeOut(hint))
        self.wait(3)

        self.play(FadeOut(formula), FadeOut(title))
        self.wait()


class Taylor(Scene):
    def construct(self):
        # Create the title components
        title1 = Text("Why is")
        title2 = MathTex(
            r"\cos(x) = \sum_{n=0}^\infty (-1)^n \frac{x^{2n}}{(2n)!} \quad \text{and} \quad "
            r"\sin(x) = \sum_{n=0}^\infty (-1)^n \frac{x^{2n+1}}{(2n+1)!}",
            font_size=36,
        )

        # Arrange the titles
        title1.to_edge(UP, buff=0.5)
        title2.next_to(title1, DOWN, buff=0.5)

        # Display the titles
        self.play(Write(title1))
        self.play(Write(title2))
        self.wait(5)

        taylor_formula = MathTex(r"\sum_{n=0}^\infty \frac{f^{(n)} (a)}{n!} (x-a)^n")
        taylor_formula.next_to(title2, DOWN, buff=0.5)
        self.play(FadeIn(taylor_formula))
        self.wait(3)
        self.play(FadeOut(taylor_formula))

        # Fade out the titles
        self.play(FadeOut(title1), FadeOut(title2))

        top = MathTex(r"\cos(x) \text{ via complex numbers}")
        top.move_to(ORIGIN)
        self.play(Write(top))
        self.wait(2)

        # Animate the expression moving to the top edge
        self.play(top.animate.to_edge(UP))
        self.wait(1)

        de_moivres_formula = MathTex(
            r"&\quad \cos(n\theta) + i \sin(n\theta) = (\cos(\theta) + i \sin(\theta))^n \\",
            r"&\quad \cos(n\theta) - i \sin(n\theta) = (\cos(\theta) - i \sin(\theta))^n ",
            font_size=48,
        ).arrange(DOWN, aligned_edge=LEFT)

        # Center the formula on the screen
        de_moivres_formula.next_to(top, DOWN, buff=0.5)

        # Animate writing the formula
        self.play(Write(de_moivres_formula))
        self.wait(5)

        dm2 = MathTex(
            r"2 \cos (n\theta) = (\cos(\theta) + i\sin(\theta))^n + (\cos(\theta) - i\sin(\theta))^n"
        )
        dm2.next_to(top, DOWN, buff=0.5)
        self.play(Transform(de_moivres_formula, dm2))
        self.wait(3)

        dm3 = MathTex(
            r"\cos (n\theta) = \frac{1}{2} (\cos(\theta) + i\sin(\theta))^n + (\cos(\theta) - i\sin(\theta))^n"
        )
        dm3.next_to(top, DOWN, buff=0.5)
        self.play(FadeOut(de_moivres_formula))
        self.wait(1)
        self.play(Write(dm3))
        self.wait(3)

        binom = MathTex(r"(x+y)^n = \sum_{k=0}^{n} \binom{n}{k}x^{n-k}y^{k}")
        binom.next_to(dm3, DOWN, buff=0.5)
        self.play(FadeIn(binom))
        self.wait(4)
        self.play(FadeOut(binom))

        dm4 = MathTex(
            r"= \frac{1}{2}\left[\cos^n (\theta) + \frac{ni\cos^{n-1}(\theta)\sin(\theta)}{1} - \frac{n(n-1)\cos^{n-2}(\theta)\sin^{2}(\theta)}{2!} - \frac{n(n-1)(n-2)i\cos^{n-3}(\theta)\sin^{3}(\theta)}{3!} + \ldots\right] \\ + \frac{1}{2}\left[\cos^n (\theta) - \frac{ni\cos^{n-1}(\theta)\sin(\theta)}{1} - \frac{n(n-1)\cos^{n-2}(\theta)\sin^{2}(\theta)}{2!} + \frac{n(n-1)(n-2)i\cos^{n-3}(\theta)\sin^{3}(\theta)}{3!} + \ldots\right]",
            font_size=28,
        )
        dm4.next_to(dm3, DOWN, buff=0.5)
        self.play(Write(dm4))
        self.wait(5)

        dm5 = MathTex(
            r"= \cos^n (\theta)- \frac{n(n-1) \cos^{n-2}(\theta) \sin^2(\theta)}{2!}+\frac{n(n-1)(n-2)(n-3)\cos^{n-4}(\theta)\sin^4 (\theta)}{4!} - \ldots",
            font_size=33,
        )
        dm5.next_to(dm2, DOWN, buff=0.75)
        self.play(Transform(dm4, dm5))
        self.wait(5)

        note1 = MathTex(
            r"\text{Let } x = n\theta, \text{ as } n \to \infty, \theta = \frac{x}{n} \to 0"
        )
        note1.next_to(dm5, DOWN, buff=0.5)
        self.play(FadeIn(note1))
        self.wait(3)
        self.play(FadeOut(note1))

        note2 = MathTex(
            r"\lim_{\theta \to 0} \cos\theta=1, \quad \lim_{\theta \to 0} \frac{\sin \theta}{\theta} = 1"
        )
        note2.next_to(dm5, DOWN, buff=0.5)
        self.play(FadeIn(note2))
        self.wait(3)
        self.play(FadeOut(note2))

        note3 = MathTex(
            r"\text{As } n \to \infty, n-1,n-2,n-3,\ldots \text{ can be replaced by } n"
        )
        note3.next_to(dm5, DOWN, buff=0.5)
        self.play(FadeIn(note3))
        self.wait(5)
        self.play(FadeOut(note3))

        dm6 = MathTex(
            r"\cos(x)= 1^n - \frac{n \cdot n \cdot (1)^{n-2} \left(\frac{x}{n}\right)^2}{2!} + \frac{n \cdot n \cdot n \cdot n \cdot (1)^{n-4} \left(\frac{x}{n}\right)^4}{4!} - \ldots",
            font_size=36,
        )
        dm6.next_to(dm5, DOWN, buff=0.75)
        self.play(Write(dm6))
        self.wait(5)

        dm7 = MathTex(
            r"\cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \ldots"
        )
        dm7.next_to(dm6, DOWN, buff=0.5)
        self.play(Write(dm7))
        self.wait(3)

        dm8 = MathTex(r"\cos(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n}}{(2n)!}")
        dm8.next_to(dm6, DOWN, buff=0.5)
        self.play(Transform(dm7, dm8))
        self.wait(5)


class EulersFormula(Scene):
    def construct(self):
        # Create and display the title
        title = MathTex(
            r"\text{Euler's Formula: } e^{ix} = \cos(x) + i\sin(x)", font_size=48
        )
        self.play(Write(title))
        self.wait(2)

        # Animate the title moving to the top edge
        self.play(title.animate.to_edge(UP))
        self.wait(1)

        let1 = MathTex(
            r"\text{Let } y= \sin(x) \implies x = \arcsin(y) = \int \frac{dy}{\sqrt{1-y^2}}"
        )
        let1.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(let1))
        self.wait(3)

        let2 = MathTex(r"\text{Let }  y= iz \implies dy=idz")
        let2.next_to(let1, DOWN, buff=0.5)
        self.play(FadeIn(let2))
        self.wait(3)
        self.play(FadeOut(let1), FadeOut(let2))

        int1 = MathTex(r"x = \arcsin(y) = \int \frac{dy}{\sqrt{1-y^2}}")
        int1.next_to(title, DOWN, buff=0.5)
        self.play(Write(int1))
        self.wait(4)

        int2 = MathTex(r"x = \int \frac{i dz}{\sqrt{1-(iz)^2}}")
        int2.next_to(title, DOWN, buff=0.5)
        self.play(Transform(int1, int2))
        self.wait(3)

        int3 = MathTex(r"x = \ i \int \frac{dz}{\sqrt{1+z^2}}")
        int3.next_to(title, DOWN, buff=0.5)
        self.play(Transform(int1, int3))
        self.wait(3)

        int4 = MathTex(r"x = i \ln(\sqrt{1+z^2} +z) +C")
        int4.next_to(int3, DOWN, buff=0.5)
        self.play(Write(int4))
        self.wait(3)

        int4_1 = MathTex(r"x = i \ln(\sqrt{1+z^2} +z)")
        int4_1.next_to(int3, DOWN, buff=0.5)
        self.play(Transform(int4, int4_1))
        self.wait(3)

        let3 = MathTex(
            r"\text{Since } z = \frac{y}{i} = \frac{\sin(x)}{i} \implies z^2 = \frac{\sin^2(x)}{i^2} = -\sin^2(x)"
        )
        let3.next_to(int4, DOWN, buff=0.5)
        self.play(FadeIn(let3))
        self.wait(3)

        self.play(FadeOut(let3))

        int5 = MathTex(r"i \ln\left(\sqrt{1-\sin^2(x)}+ \frac{\sin(x)}{i}\right)")
        int5.next_to(int4, DOWN, buff=0.5)
        self.play(Write(int5))
        self.wait(5)

        int6 = MathTex(r"x = i \ln(\cos(x) - i\sin(x))")
        int6.next_to(int4, DOWN, buff=0.5)
        self.play(Transform(int5, int6))
        self.wait(3)

        mult = MathTex(r"ix = i^2 \ln(\cos(x) - i\sin(x))")
        mult.next_to(int6, DOWN, buff=0.5)
        self.play(Write(mult))
        self.wait(3)

        mult2 = MathTex(r"ix = - \ln(\cos(x) - i\sin(x))")
        mult2.next_to(int6, DOWN, buff=0.5)
        self.play(Transform(mult, mult2))
        self.wait(3)

        mult3 = MathTex(r"ix = \ln \left(\frac{1}{\cos(x)-i\sin(x)}\right)")
        mult3.next_to(int6, DOWN, buff=0.5)
        self.play(Transform(mult, mult3))
        self.wait(3)

        mult4 = MathTex(r"ix = \ln(\cos(x) +i\sin(x))")
        mult4.next_to(int6, DOWN, buff=0.5)
        self.play(Transform(mult, mult4))
        self.wait(3)

        final = MathTex(r"e^{ix} = e^{\ln(\cos(x) + i \sin(x))}")
        final.next_to(int6, DOWN, buff=0.5)
        self.play(Transform(mult, final))
        self.wait(2)

        final2 = MathTex(r"e^{ix} = \cos(x) + i \sin(x)")
        final2.next_to(final, DOWN, buff=0.5)
        self.play(Write(final2))
        self.wait(7)


class CoolPi(Scene):
    def construct(self):
        # Create and display the title
        eulerformula = MathTex(r"e^{ix} = \cos(x) + i\sin(x)", font_size=72)
        eulerformula.move_to(ORIGIN)
        self.play(Write(eulerformula))
        self.wait(2)

        eulerformula2 = MathTex(r"e^{i\pi} = \cos(\pi) + i\sin(\pi)", font_size=72)
        eulerformula2.move_to(ORIGIN)
        self.play(Transform(eulerformula, eulerformula2))
        self.wait(3)

        eulerformula3 = MathTex(r"e^{i\pi} = -1 + i \cdot 0", font_size=72)
        eulerformula3.move_to(ORIGIN)
        self.play(Transform(eulerformula, eulerformula3))
        self.wait(3)

        eulerformula4 = MathTex(r"e^{i\pi} = -1 ", font_size=72)
        eulerformula4.move_to(ORIGIN)
        self.play(Transform(eulerformula, eulerformula4))
        self.wait(3)

        eulerformula5 = MathTex(r"e^{i\pi} +1 = 0 ", font_size=72)
        eulerformula5.move_to(ORIGIN)
        self.play(Transform(eulerformula, eulerformula5))
        self.wait(10)


class Exercises(Scene):
    def construct(self):
        # Create and display the title
        title = Text("Exercises", font_size=48)
        self.play(Write(title))
        self.wait(2)

        # Animate the title moving to the top edge
        self.play(title.animate.to_edge(UP))
        self.wait(1)

        # Create bullet points
        bullet_point1 = MathTex(
            r"\text{1. Show } e^{ix} = \cos(x) + i\sin(x) \text{ by Taylor Series}",
            font_size=48,
        )
        bullet_point2 = MathTex(
            r"\text{2. Find } i^i ",
            font_size=48,
        )
        bullet_point3 = MathTex(
            r"\text{3. Bernoulli believed that} \ln(-x) = \ln(x) \text{ is true. Disprove this.}",
            font_size=48,
        )
        bullet_point4 = MathTex(
            r"\text{What 'factor' does } \ln(-x) \text{ and } \ln(x) \text{ differ by?}",
            font_size=48,
        )

        bullet_point5 = MathTex(
            r"\text{(final answer not be in terms of a logarithm)}",
            font_size=48,
        )

        # Group the bullet points vertically and align them
        bullet_points = VGroup(
            bullet_point1, bullet_point2, bullet_point3, bullet_point4, bullet_point5
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        # Position the bullet points below the title
        bullet_points.move_to(ORIGIN)

        # Display the bullet points sequentially
        for bullet in bullet_points:
            self.play(Write(bullet))
            self.wait(3)

        self.wait(2)
