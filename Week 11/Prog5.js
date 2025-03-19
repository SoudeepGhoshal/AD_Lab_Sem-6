class Order {
    #burgerPrice = 5.99;
    #friesPrice = 2.49;

    constructor(numBurgers, numFries) {
        this.burgers = numBurgers;
        this.fries = numFries;
    }

    calculateTotal() {
        return (this.burgers * this.#burgerPrice) + (this.fries * this.#friesPrice);
    }

    get totalCost() {
        return this.calculateTotal();
    }
}

const order1 = new Order(2, 3);
const order2 = new Order(1, 0);
const order3 = new Order(0, 5);

console.log(`Order 1 total: $${order1.totalCost.toFixed(2)}`);
console.log(`Order 2 total: $${order2.totalCost.toFixed(2)}`);
console.log(`Order 3 total: $${order3.totalCost.toFixed(2)}`);