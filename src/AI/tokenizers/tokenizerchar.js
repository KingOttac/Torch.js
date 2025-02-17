function tokenizerchar(lines,type) {

	tokens[0] = "\n";
	let convertedlines = [];
	for (a = 0; convertedlines.length < int(type)*(learningset+sampleset); a++) {
		convertedlines = concatenate([convertedlines,split(lines[a] + "\n","")]);
	}
	convertedlines = convertedlines.slice(0,int(type)*(learningset+sampleset));
	let newcl = [];
	for (a = 0; a < convertedlines.length; a += int(type)) {
		newcl[a/int(type)] = join(convertedlines.slice(a,a+int(type)),"");
		if (untoken(newcl[a/int(type)]) == -1) {
			tokens[tokens.length] = newcl[a/int(type)];
		}
	}
	convertedlines = newcl;

	for (a = 0; a < convertedlines.length; a++) {
		convertedlines[a] = untoken(convertedlines[a]);
	}//convert everything into numbers
	return convertedlines;
	
}//converts one data file (from preload) into tokens in the final array
