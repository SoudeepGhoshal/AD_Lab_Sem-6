let inventory = [];

const item1 = {
    name: "Laptop",
    model: "XPro-15",
    cost: 1299.99,
    quantity: 25
};

const item2 = {
    name: "Smartphone",
    model: "Galaxy-Z",
    cost: 799.99,
    quantity: 50
};

const item3 = {
    name: "Headphones",
    model: "AudioMax-300",
    cost: 149.99,
    quantity: 100
};

inventory.push(item1, item2, item3);

console.log("Complete Inventory:", inventory);
console.log("Quantity of third item (Headphones):", inventory[2].quantity);

const item4 = {
    name: "Tablet",
    model: "Tab-S8",
    cost: 499.99,
    quantity: 35
};
inventory.push(item4);

function getItemByName(itemName) {
    return inventory.find(item => item.name === itemName);
}

function getTotalInventoryValue() {
    return inventory.reduce((total, item) => total + (item.cost * item.quantity), 0);
}

function getItemsBelowQuantity(threshold) {
    return inventory.filter(item => item.quantity < threshold);
}

console.log("\nAdditional Queries:");
console.log("Laptop details:", getItemByName("Laptop"));
console.log("Total inventory value: $", getTotalInventoryValue().toFixed(2));
console.log("Items with quantity below 40:", getItemsBelowQuantity(40));

console.log("\nAccessing specific elements:");
console.log("Second item's model:", inventory[1].model);
console.log("Fourth item's cost:", inventory[3].cost);
console.log("First item's quantity:", inventory[0].quantity);