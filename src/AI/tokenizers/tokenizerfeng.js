function tokenizerfeng(lines,type) {

	tokens[0] = "<spac>";
	tokens[1] = "<EOS>";
	for (let a = 0; a < 2*sampleset; a++) {
		let toadd = split(lines[a],"");
		lines[a] = toadd;
	}
	let newcl = maketensor(2,[lines.length]);
	for (let a = 0; a < lines.length; a++) {
		for (let b = 0; b < lines[a].length; b += int(type)) {
			newcl[a][b/int(type)] = join(lines[a].slice(b,min(b+int(type),lines[a].length)),"");
			if (untoken(newcl[a][b/int(type)]) == -1) {
				tokens[tokens.length] = newcl[a][b/int(type)];
			}
			newcl[a][b/int(type)] = untoken(newcl[a][b/int(type)]);
		}
		if (newcl[a].length+1 > learningset) {
			learningset = newcl[a].length+1;
		}
	}
	lines = [];
	for (let a = 0; a < newcl.length; a += 2) {
		lines[a/2] = [newcl[a],newcl[a+1]];
	}

	return lines;
	
}
