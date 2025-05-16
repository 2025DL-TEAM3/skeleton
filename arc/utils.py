# system prompt
system_prompt = (
    "You are an expert at solving puzzles from the Abstraction and Reasoning Corpus (ARC). "
    "From a few input/output examples, infer the transformation rule "
    "and apply it to a new test grid."
)

# user prompt 1: examples
user_message_template1 = (
    "Here are {n} example pair{plural}:\n"
    "{examples}\n"
    "Observe how each input becomes its output."
)

# user prompt 2: test input
user_message_template2 = (
    "Now apply that rule to this test input grid:\n"
    "{test_grid}"
)

# user prompt 3: output format
user_message_template3 = (
    "Only return the output grid (rows as digit sequences; each ending with a newline; no extra text or spaces):"
)