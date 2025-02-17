function tokenizerfeng(lines,type) {
	
	tokens[0] = "<spac>";
	for (a = 0; a < 2*sampleset; a++) {
		let toadd = split(lines[a],"");
		lines[a] = toadd;
	}
	let newcl = maketensor(1,[lines.length],[]);
	for (a = 0; a < lines.length; a++) {
		for (b = 0; b < lines[a].length; b += int(type)) {
			newcl[a][b/int(type)] = join(lines[a].slice(b,min(b+int(type),lines[a].length)),"");
			if (untoken(newcl[a][b/int(type)]) == -1) {
				tokens[tokens.length] = newcl[a][b/int(type)];
			}
		}
	}
	lines = [];
	for (a = 0; a < newcl.length; a += 2) {
		lines[a/2] = [newcl[a],newcl[a+1]];
	}
	for (a = 0; a < lines.length; a++) {
		for (b = 0; b < lines[a].length; b++) {
			if (lines[a][b].length > learningset) {
				learningset = lines[a][b].length;
			}//find greatest learningset
			for (c = 0; c < lines[a][b].length; c++) {
				lines[a][b][c] = untoken(lines[a][b][c]);
			}
		}
	}

	return lines;
	
}
