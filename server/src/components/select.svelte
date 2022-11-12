<script lang="ts">
	import { Grid, Row, Column, Tile, ImageLoader, SelectableTile } from 'carbon-components-svelte';

	export let images: {
		name: string;
		url: string;
		pollen?: { box: { x: number; y: number; w: number; h: number } };
	}[] = [];

	let selectedIndex = 0;
</script>

<Grid>
	<Row>
		<Column>
			<Tile>
				<img
					src={images[selectedIndex].url}
					style="width: 100%;"
					alt={images[selectedIndex].name}
				/>
				<Column padding>
					<p>{images[selectedIndex].name}</p>
				</Column>
			</Tile>
		</Column>
	</Row>
</Grid>
<br />
<br />
<Grid>
	<Row padding>
		<div class="container">
			{#each images as image, i}
				<SelectableTile
					style="display: inline-block;"
					selected={selectedIndex === images.indexOf(image)}
					on:click={() => {
						selectedIndex = i;
					}}
				>
					<img src={image.url} alt={image.name} style="width: 10rem;" />
				</SelectableTile>
			{/each}
		</div>
	</Row>
</Grid>

<style>
	.container {
		width: 100%;
		overflow-x: auto;
		white-space: nowrap;
	}
</style>
