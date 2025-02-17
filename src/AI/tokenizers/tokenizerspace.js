function tokenizerspace(lines) {

	tokens[0] = "\n";
	let convertedlines = [];
	for (a = 0; convertedlines.length < learningset+sampleset && a < lines.length; a++) {
		
		let listarr = [];
		listarr = lines[a];
		listarr = split(listarr," ");
		for (b = 0; b < listarr.length-1; b++) {
			listarr[b] += " ";
		}
		if (listarr[0] != "") {
			listarr[listarr.length] = "\n";
		}
		else {
			listarr[0] = "\n";
		}
		for (b = 0; b < listarr.length && convertedlines.length < learningset+sampleset; b++) {
			convertedlines[convertedlines.length] = listarr[b];
			if (untoken(listarr[b]) == -1) {
				tokens[tokens.length] = listarr[b];
			}
		}
		
	}

	for (a = 0; a < convertedlines.length; a++) {
		convertedlines[a] = untoken(convertedlines[a]);
	}//convert everything into numbers
	
	return convertedlines;
	
}//converts one data file (from preload) into tokens in the final array
