function tokenizer(lines,type) {

	let convertedlines = [];
	for (a = 0; a < sampleset; a++) {
		
		let listarr = [];
		if (type == "space") {
			listarr = lines[a] + " \n";
			listarr = split(listarr," ");
		}
		else if (type == "char") {
			listarr = lines[a] + "\n";
			listarr = split(listarr,"");
		}
		for (b = 0; b < listarr.length; b++) {
			if (listarr[b] != "\n" && type == "space") {
				listarr[b] += " ";
			}
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
