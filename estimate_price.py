def parse_coeficients(file):
    coeficients = []
    with open(file) as f:
        for line in f:
            values = line.split('\t')
            coeficients += [float(value) for value in values]
    if len(coeficients) != 2:
        raise ValueError
    return coeficients


def compute_price(coeficients, millage):
    price = coeficients[0] + millage*coeficients[1]
    return price

def main():
    try:
        coeficients = parse_coeficients("./.coefficients.txt")
    except ValueError:
        print("Invalid Coefficients file, train the model or check .coeficients.txt")

    millage = None
    message = "Enter car millage: "
    while millage is None:
        try:
            millage = float(input(message))
            if millage < 0:
                raise ValueError
        except:
            millage = None
            message = "Millage must be a positive number: "
    print(f'The price is: {compute_price(coeficients, millage):.2f}')

if __name__ == "__main__":
    main()
