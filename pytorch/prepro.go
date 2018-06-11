package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
)

func main() {
	set := make(map[rune]int)
	key := make([]rune, 0)
	total := 0
	buf := new(bytes.Buffer)

	fk, err := os.Create("key.txt")
	if err != nil {
		fmt.Printf("create file: %v\n", err)
		return
	}
	defer fk.Close()
	fd, err := os.Create("poem.dat")
	if err != nil {
		fmt.Printf("create file: %v\n", err)
		return
	}
	defer fd.Close()

	fl, err := getFileList("../../data/chinese-poetry/json/")
	if err != nil {
		fmt.Printf("get file list: %v\n", err)
		return
	}
	fmt.Printf("Sourcing %d files, ", len(fl))

	for _, filename := range fl {
		fp, err := os.Open(filename)
		if err != nil {
			fmt.Printf("open file: %v\n", err)
			return
		}
		defer fp.Close()

		decoder := json.NewDecoder(fp)
		pm := make([]*poetry, 0)
		if err := decoder.Decode(&pm); err != nil {
			fmt.Printf("decode json: %v\n", err)
			return
		}

		for _, p := range pm {
			bpm := new(bytes.Buffer)
			for _, n := range p.Paragraphs {
				for _, c := range n {
					var ck int
					if i, ok := set[c]; !ok {
						ck = len(key)
						key = append(key, c)
						set[c] = ck
					} else {
						ck = i
					}
					binary.Write(bpm, binary.BigEndian, int16(ck))
				}
			}
			binary.Write(buf, binary.BigEndian, int16(bpm.Len()/2))
			bpm.WriteTo(buf)
		}
		total += len(pm)
	}
	binary.Write(fd, binary.BigEndian, int32(total))
	buf.WriteTo(fd)

	for i, k := range key {
		if i == 0 {
			fmt.Fprintf(fk, "%c", k)
		} else {
			fmt.Fprintf(fk, "\n%c", k)
		}
	}
	fmt.Printf("total %d poems readed.\n", total)
}

func getFileList(dataDir string) ([]string, error) {
	if len(os.Args[1:]) != 1 {
		return nil, fmt.Errorf("missing data file pattern")
	}
	fs, err := ioutil.ReadDir(dataDir)
	if err != nil {
		return nil, err
	}
	fl := make([]string, 0)
	for _, fi := range fs {
		if ok, err := regexp.MatchString(os.Args[1], fi.Name()); err == nil && ok && !fi.IsDir() {
			fl = append(fl, dataDir+fi.Name())
		}
	}
	return fl, nil
}

type poetry struct {
	Title      string
	Author     string
	Paragraphs []string
	Strains    []string
}
