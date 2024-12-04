function tokenizer(lines,type) {

	let convertedlines = [];
	for (a = 0; convertedlines.length < learningset+sampleset; a++) {
		
		let listarr = [];
		if (type == "space") {
			listarr = lines[a];
			listarr = split(listarr," ");
			let rlistarr = [];
			for (b = 0; b < listarr.length; b++) {
				rlistarr[rlistarr.length] = listarr[b];
				if (b != listarr.length-1) {
					rlistarr[rlistarr.length] = " ";
				}
				else {
					rlistarr[rlistarr.length] = "\n";
				}
			}
			listarr = rlistarr;
		}
		else if (type == "char") {
			listarr = lines[a] + "\n";
			listarr = split(listarr,"");
		}
		for (b = 0; b < listarr.length && convertedlines.length < learningset+sampleset; b++) {
			convertedlines[convertedlines.length] = listarr[b];
			if (untoken(listarr[b]) == -1) {
				tokens[tokens.length] = listarr[b];
			}
		}
		
	}

	for (a = 0; a < convertedlines.length; a++) {
		convertedlines[a] = untoken(convertedlines[a])
	}//convert everything into numbers
	return convertedlines;
	
}//converts one data file (from preload) into tokens in the final array
