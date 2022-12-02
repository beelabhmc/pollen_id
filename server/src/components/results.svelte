<script lang="ts">
	import { Grid, Row, Column, DataTable, Button } from 'carbon-components-svelte';
	import Download from 'carbon-icons-svelte/lib/Download.svelte';

	export let images: {
		name: string;
		img: HTMLImageElement;
		pixels_per_micron: number;
		pollen: { species?: [string, number][]; box: { x: number; y: number; w: number; h: number } }[];
	}[] = [];
	export let topN: string = '1';

	let rows: Array<string | number>[] = [];
	let headerRow = ['id', 'filename', 'x', 'y', 'w', 'h'];
	$: {
		headerRow = ['id', 'filename', 'x', 'y', 'w', 'h'];
		for (let i = 0; i < Number(topN); i++) {
			headerRow = [...headerRow, `species${i + 1}`, `confidence${i + 1}`];
		}
	}

	$: {
		rows = [];
		for (let i = 0; i < images.length; i++) {
			for (let j = 0; j < images[i].pollen.length; j++) {
				let row = [
					i + '-' + j,
					images[i].name,
					images[i].pollen[j].box.x,
					images[i].pollen[j].box.y,
					images[i].pollen[j].box.w,
					images[i].pollen[j].box.h
				];
				if (images[i].pollen[j].species) {
					images[i].pollen[j].species?.forEach((species) => {
						row = [...row, ...species];
					});
				} else {
					for (let k = 0; k < Number(topN); k++) {
						row = [...row, '', 0];
					}
				}
				rows.push(row);
			}
		}
	}

	function saveCSV() {
		let csvContent =
			'data:text/csv;charset=utf-8,' + headerRow + '\n' + rows.map((e) => e.join(',')).join('\n');
		const encodedUri = encodeURI(csvContent);
		window.open(encodedUri);
	}
</script>

<Grid>
	<Row>
		<Column padding>
			<DataTable
				stickyHeader
				size="short"
				title="Detected Pollen"
				headers={headerRow.map((header) => ({ key: header, value: header }))}
				rows={rows.map((row) => {
					let r = {
						id: row[0],
						filename: row[1],
						x: row[2],
						y: row[3],
						w: row[4],
						h: row[5]
					};
					for (let i = 0; i < Number(topN); i++) {
						r = { ...r, [`species${i + 1}`]: row[6 + i * 2], [`confidence${i + 1}`]: row[7 + i * 2].toFixed(3) };
					}
					return r;
				})}
			/>
		</Column>
	</Row>
	<Row>
		<Button icon={Download} on:click={saveCSV}>Download</Button>
	</Row>
</Grid>
