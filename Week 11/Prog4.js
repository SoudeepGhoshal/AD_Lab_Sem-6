class Employee {
    constructor(firstName, lastName, yearsWorked) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.yearsWorked = yearsWorked;
    }

    getDetails() {
        return `${this.firstName} ${this.lastName} has worked at the company for ${this.yearsWorked} years`;
    }
}

const employees = [
    new Employee("John", "Smith", 5),
    new Employee("Maria", "Gonzalez", 3),
    new Employee("Alex", "Johnson", 8)
];

employees.forEach(employee => {
    console.log(employee.getDetails());
});