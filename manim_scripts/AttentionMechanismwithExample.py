from manim import *

class AttentionMechanismWithExamples(Scene):
    def construct(self):
        self.show_title()
        self.introduction()
        self.show_analogy()
        self.plot_keys_and_query()
        self.show_matrices()
        self.calculate_dot_products()
        self.show_attention_formula1()
        self.scale_and_softmax()
        self.show_attention_formula2()
        self.compute_weighted_sum()
        self.show_attention_formula3()

    def show_title(self):
        """Displays the title of the scene."""
        title = Text("Visualizing Attention Mechanisms", font_size=36).scale(1)
        self.play(Write(title))
        self.wait(3)
        self.play(FadeOut(title))

    def introduction(self):
        """Introduces the scenario for the attention mechanism."""
        intro_text = Text("Imagine you have three groups at party discussing:", font_size=24).shift(UP*2.5)
        intro_text2 = Text("Sports, Movies, and Work.").next_to(intro_text, DOWN, buff=0.5)
        intro_text3 = Text("And you are more interested in Cricket", font_size=24).next_to(intro_text2, DOWN, buff=0.5)
        
        self.play(Write(intro_text))
        self.play(Write(intro_text2))
        self.play(Write(intro_text3))
        self.wait(2)
        self.play(FadeOut(intro_text), FadeOut(intro_text2), FadeOut(intro_text3))

    def show_analogy(self):
        """Explains key, query, and value with an analogy."""
        analogy_text = Text("In this context, 'Keys' are different groups,", font_size=24).shift(UP*2.5)
        analogy_text2 = Text("the 'Query' is your interest, and 'Values' represent", font_size=24).next_to(analogy_text, DOWN, buff=0.5)
        analogy_text3 = Text("actual content in those conversations.", font_size=24).next_to(analogy_text2, DOWN, buff=0.5)
        
        self.play(Write(analogy_text))
        self.play(Write(analogy_text2))
        self.play(Write(analogy_text3))
        self.wait(2)
        self.play(FadeOut(analogy_text), FadeOut(analogy_text2), FadeOut(analogy_text3))

    def plot_keys_and_query(self):
        """Plots the keys and query on a 2D graph and returns the axes and query."""
        axes = Axes(
            x_range=[0, 1.2, 0.2], y_range=[0, 1.2, 0.2],
            axis_config={"color": BLUE},
            x_length=6, y_length=6,
            tips=False
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Define Keys and their labels
        key1 = Dot(axes.coords_to_point(0.9, 0.6), color=RED)
        key2 = Dot(axes.coords_to_point(0.2, 0.1), color=GREEN)
        key3 = Dot(axes.coords_to_point(0.1, 0.3), color=ORANGE)
        
        key1_label = Text("K1 (Sports)",font_size=16).next_to(key1, UP)
        key2_label = Text("K2 (Movies)",font_size=16).next_to(key2, UP)
        key3_label = Text("K3 (Work)",font_size=16).next_to(key3, UP)
        
        # Define Query and its label
        query = Dot(axes.coords_to_point(1, 0.5), color=YELLOW)
        query_label = Text("Q (Cricket)", font_size=16).next_to(query, DOWN)
        
        self.play(Create(axes), Write(labels))
        self.play(FadeIn(key1, key1_label))
        self.play(FadeIn(key2, key2_label))
        self.play(FadeIn(key3, key3_label))

        self.play(FadeIn(query, query_label))
        self.wait(3)
        """Draws the projection lines for dot products."""
        proj1 = DashedLine(query.get_center(), axes.coords_to_point(0.9, 0.6), color=RED, dash_length=0.1)
        proj2 = DashedLine(query.get_center(), axes.coords_to_point(0.2, 0.1), color=GREEN, dash_length=0.1)
        proj3 = DashedLine(query.get_center(), axes.coords_to_point(0.1, 0.3), color=ORANGE, dash_length=0.1)

        self.play(Create(proj2))
        self.play(Create(proj3))
        self.play(Create(proj1))
        self.wait(3)
        self.play(FadeOut(axes), FadeOut(labels), FadeOut(query), FadeOut(key1), FadeOut(key2), FadeOut(key3), 
                  FadeOut(proj1), FadeOut(proj2), FadeOut(proj3),
                  FadeOut(key1_label), FadeOut(key2_label), FadeOut(key3_label), FadeOut(query_label))

    def show_matrices(self):
        """Displays the matrices for Query, Keys, and Values."""
        intro_text = Text("Let’s put Query, Key, and Value in matrices.", font_size=24).shift(UP*3)
        self.play(Write(intro_text))

        query_matrix = Matrix([
            [1.0, 0.5]
        ], element_to_mobject_config={"color": RED}).shift(LEFT*5)

        key_matrix = Matrix([
            [0.9, 0.6],
            [0.2, 0.1],
            [0.1, 0.3]
        ], element_to_mobject_config={"color": GREEN}).shift(LEFT*1)

        value_matrix = Matrix([
            [10.0, 0.0],
            [0.0, 10.0],
            [5.0, 5.0]
        ], element_to_mobject_config={"color": ORANGE}).shift(RIGHT*5)

        query_label = Text("Query(Q)", font_size=20).next_to(query_matrix, DOWN)
        key_label = Text("Keys(K)", font_size=20).next_to(key_matrix, DOWN)
        value_label = Text("Values(V)", font_size=20).next_to(value_matrix, DOWN)

        self.play(Write(query_matrix), Write(query_label))
        self.wait(1)
        self.play(Write(key_matrix), Write(key_label))
        self.wait(1)
        self.play(Write(value_matrix), Write(value_label))
        self.wait(2)
        self.play(FadeOut(intro_text))
        self.play(FadeOut(query_matrix), FadeOut(key_matrix), FadeOut(value_matrix),
                  FadeOut(query_label), FadeOut(key_label), FadeOut(value_label))

    def calculate_dot_products(self):
        """Calculates and shows the dot products (scores)."""
        score_text = Text("Let's Match Interest (Dot Product)").shift(UP*3)
        dot_products = MathTex(
            r"Q \cdot K_1^T (Sports) = 1 \times 0.9 + 0.5 \times 0.6 = 1.2", 
            r"Q \cdot K_2^T (Movies) = 1 \times 0.2 + 0.5 \times 0.1 = 0.25",
            r"Q \cdot K_3^T (Work)   = 1 \times 0.1 + 0.5 \times 0.3 = 0.25"
        ).arrange(DOWN)

        self.play(Write(score_text))
        self.play(Write(dot_products))
        self.wait(3)
        self.play(FadeOut(score_text), FadeOut(dot_products))

    def scale_and_softmax(self):
        """Applies scaling and softmax to the scores."""
        scal_text = Text("Preventing Extremes (Scale Scores)").shift(UP*3)
        scaled_scores = MathTex(
            r"\frac{[1.2, 0.25, 0.25]}{\sqrt{2}} = [0.85, 0.18, 0.18]"
        ).arrange(DOWN)

        self.play(Write(scal_text), Write(scaled_scores))
        self.wait(3)
        self.play(FadeOut(scaled_scores))
        self.play(FadeOut(scal_text))

        softmax_text = Text("Put Scores into Probabilities (Softmax)").shift(UP*3)
        softmax = MathTex(
            r"\text{Softmax}([0.85, 0.18, 0.18]) = [0.61, 0.19, 0.19]"
        ).arrange(DOWN)

        self.play(Write(softmax_text))
        self.play(Write(softmax))
        self.wait(3)
        self.play(FadeOut(softmax), FadeOut(softmax_text))

    def compute_weighted_sum(self):
        """Computes and displays the weighted sum of the values."""
        final_text = Text("Final Content that you heard").shift(UP*3)
        weighted_sum = MathTex(
            r"\text{Attention Output} = 0.61 \times V_1 + 0.19 \times V_2 + 0.19 \times V_3"
        ).next_to(final_text, DOWN)
        final_output = MathTex(
            r"= [7.05, 2.85]"
        ).next_to(weighted_sum, DOWN, buff=1)
        summary = Text("Out of three groups, you mostly listened to sports (Closer to Sports Value [10.0, 0.0])", font_size=24).next_to(final_output, DOWN, buff=1)

        self.play(Write(final_text))
        self.play(Write(weighted_sum))
        self.play(Write(final_output))
        self.wait(1)
        self.play(Write(summary))
        self.wait(3)
        self.play(FadeOut(weighted_sum), FadeOut(final_output), FadeOut(summary), FadeOut(final_text))

    def show_attention_formula1(self):
        """Calculation Till Dot Product"""
        atten_text = Text("Calculation Till Now").shift(UP*3)
        attention_formula = MathTex(
            r"{QK^T}"
        ).next_to(atten_text, DOWN, buff=1)

        self.play(Write(atten_text))
        self.play(Write(attention_formula))
        self.wait(3)
        self.play(FadeOut(atten_text),FadeOut(attention_formula))

    def show_attention_formula2(self):
        """Calculation Till Scale and Softmax."""
        atten_text = Text("Calculation Till Now").shift(UP*3)
        attention_formula = MathTex(
            r"\text{Attention}(Q, K) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)"
        ).next_to(atten_text, DOWN, buff=1)

        self.play(Write(atten_text))
        self.play(Write(attention_formula))
        self.wait(3)
        self.play(FadeOut(atten_text),FadeOut(attention_formula))

    def show_attention_formula3(self):
        """Displays the final attention formula."""
        atten_text = Text("Final Attention Formula").shift(UP*3)
        attention_formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
        ).next_to(atten_text, DOWN, buff=1)

        self.play(Write(atten_text))
        self.play(Write(attention_formula))
        self.wait(3)
        self.play(FadeOut(atten_text),FadeOut(attention_formula))


