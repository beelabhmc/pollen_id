<script lang="ts">
	import { Grid, Row, Column, DataTable, Button } from 'carbon-components-svelte';
	import Download from 'carbon-icons-svelte/lib/Download.svelte';

	export let images: {
		name: string;
		img: HTMLImageElement;
		pixels_per_micron: number;
		pollen: { species?: string; box: { x: number; y: number; w: number; h: number } }[];
	}[] = [];

	let rows: [string, string, string, number, number, number, number][] = [];
	const headerRow = 'id, filename, species, x, y, w, h';

	$: {
		rows = [];
		for (let i = 0; i < images.length; i++) {
			for (let j = 0; j < images[i].pollen.length; j++) {
				rows.push([
					i + '-' + j,
					images[i].name,
					images[i].pollen[j].species || 'Unknown',
					images[i].pollen[j].box.x,
					images[i].pollen[j].box.y,
					images[i].pollen[j].box.w,
					images[i].pollen[j].box.h
				]);
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
				headers={[
					{ key: 'filename', value: 'Filename' },
					{ key: 'species', value: 'Species' },
					{ key: 'x', value: 'x' },
					{ key: 'y', value: 'y' },
					{ key: 'w', value: 'w' },
					{ key: 'h', value: 'h' }
				]}
				rows={rows.map((row) => {
					return {
						id: row[0],
						filename: row[1],
						species: row[2],
						x: row[3],
						y: row[4],
						w: row[5],
						h: row[6]
					};
				})}
			/>
		</Column>
	</Row>
	<Row>
		<Button icon={Download} on:click={saveCSV}>Download</Button>
	</Row>
</Grid>
