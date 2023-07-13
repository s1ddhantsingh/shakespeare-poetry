const axios = require("axios");
const fs = require("fs");
const path = require("path");

const apiUrl = "https://spotify-lyric-api.herokuapp.com/";
const trackIdsFile = "track_ids.txt";
const jsonDir = "./responses/json";
const txtDir = "./responses/txt";

// Create the directories if they don't exist
fs.mkdirSync(jsonDir, { recursive: true });
fs.mkdirSync(txtDir, { recursive: true });

// Read track IDs from file
fs.readFile(trackIdsFile, "utf-8", (err, data) => {
	if (err) {
		console.error("Error reading track IDs file:", err);
		return;
	}

	const trackIds = data.split("\n").filter(Boolean);

	trackIds.forEach((trackId) => {
		axios
			.get(apiUrl, {
				params: {
					trackid: trackId,
				},
			})
			.then((response) => {
				const data = response.data;

				if (data.error) {
					console.error(
						`Error in API response for track ID ${trackId}:`,
						data.error
					);
					return;
				}

				const jsonFilePath = path.join(jsonDir, `${trackId}.json`);
				const txtFilePath = path.join(txtDir, `${trackId}.txt`);

				// Save JSON response to file
				const jsonContent = JSON.stringify(data, null, 2);
				fs.writeFile(jsonFilePath, jsonContent, (err) => {
					if (err) {
						console.error(
							`Error writing JSON file for track ID ${trackId}:`,
							err
						);
					} else {
						console.log(`JSON file saved for track ID ${trackId}.`);
					}
				});

				// Save lyrics to text file
				const lines = data.lines;
				const wordsLines = lines.map((line) => line.words);
				const txtContent = wordsLines.join("\n");
				fs.writeFile(txtFilePath, txtContent, (err) => {
					if (err) {
						console.error(
							`Error writing TXT file for track ID ${trackId}:`,
							err
						);
					} else {
						console.log(`TXT file saved for track ID ${trackId}.`);
					}
				});
			})
			.catch((error) => {
				console.error(
					`Error making API request for track ID ${trackId}:`,
					error
				);
			});
	});
});
