from pyparsing import Word, alphanums, Literal, Group, infixNotation, opAssoc, CaselessKeyword, ParseException

# Define grammar
integer = Word(alphanums)
variable = Word(alphanums)
comparison_op = Literal(">=") | Literal("<=") | Literal("==") | Literal("!=") | Literal(">") | Literal("<")
condition = Group(variable + comparison_op + integer)
statement = Group(variable + "=" + integer)

# Define actions
def process_condition(tokens):
    return f"{tokens[0]} {tokens[1]} {tokens[2]}"

def process_statement(tokens):
    return f"{tokens[0]} == {tokens[1]}"

condition.setParseAction(process_condition)
statement.setParseAction(process_statement)

# Parse the input string
input_string = "allHad==1;jets>=8;bjets>=1"
conditions_and_statements = input_string.split(";")


if __name__ == '__main__':
    for iter in  conditions_and_statements:
        print(iter)
        print(condition.parseString(iter))
    # parsed_conditions = [condition.parseString(cond).asList() for cond in conditions_and_statements if ">=" in cond or "<=" in cond or "==" in cond or "!=" in cond]

    # print(parsed_statements )
    # try:
    #     parsed_conditions = [condition.parseString(cond).asList() for cond in conditions_and_statements if ">=" in cond or "<=" in cond or "==" in cond or "!=" in cond]
    #     parsed_statements = [statement.parseString(stmt).asList() for stmt in conditions_and_statements if "=" in stmt]

    #     # Print parsed conditions
    #     for cond in parsed_conditions:
    #         print("Parsed condition:", cond)

    #     # Print parsed statements
    #     for stmt in parsed_statements:
    #         print("Parsed statement:", stmt)

    #     # Generate if conditions and statements
    #     for cond in parsed_conditions:
    #         print("if", cond[0], cond[1], cond[2])
    #     for stmt in parsed_statements:
    #         print(stmt[0], "=", stmt[1])

    # except ParseException as e:
    #     print("Error:", e)
    exit(0)