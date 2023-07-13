require("dotenv").config();

describe("template spec", () => {
	it("passes", () => {
		cy.visit("https://example.cypress.io");
		console.log(process.env.email);
	});
});
