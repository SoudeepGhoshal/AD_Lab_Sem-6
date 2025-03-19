const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

const choices = ["Rock", "Paper", "Scissors"];

readline.question("Enter your choice (Rock, Paper, or Scissors): ", (userInput) => {
    const playerIndex = choices.findIndex(choice => choice.toLowerCase() === userInput.trim().toLowerCase());
    const computerIndex = Math.floor(Math.random() * 3);

    if (playerIndex === -1) {
        console.log("Invalid choice! Please enter Rock, Paper, or Scissors.");
    } else {
        let responseMessage = `Player chose: ${choices[playerIndex]}\nComputer chose: ${choices[computerIndex]}`;

        let result;
        if (playerIndex === computerIndex) {
            result = "It's a tie!";
        } else if (
            (playerIndex === 0 && computerIndex === 2) || // Rock beats Scissors
            (playerIndex === 1 && computerIndex === 0) || // Paper beats Rock
            (playerIndex === 2 && computerIndex === 1)   // Scissors beats Paper
        ) {
            result = "Player wins!";
        } else {
            result = "Computer wins!";
        }

        const finalMessage = `${responseMessage}\nResult: ${result}`;
        console.log(finalMessage);
    }
    readline.close();
});