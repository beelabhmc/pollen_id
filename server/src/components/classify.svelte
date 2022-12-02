<script lang="ts">
	import {
		Grid,
		Row,
		Column,
		Tile,
		Button,
		ProgressBar,
		Form,
		FormGroup,
		RadioButtonGroup,
		RadioButton
	} from 'carbon-components-svelte';

	import ImageWithOverlay from './ImageWithOverlay.svelte';
	import Box from './box.svelte';

	export let images: {
		name: string;
		img: HTMLImageElement;
		pixels_per_micron: number;
		pollen: { species?: [string, number][]; box: { x: number; y: number; w: number; h: number } }[];
	}[] = [];

	let classifyingProgress = 0;

	async function classifyPollen() {
		await Promise.all(
			images.map(async (image, i) => {
				const imgAsBlob = await (await fetch(image.img.src)).blob();

				const formData = new FormData();
				formData.append('file', imgAsBlob, image.name);
				formData.append(
					'metadata',
					JSON.stringify({
						pixels_per_micron: image.pixels_per_micron,
						crop_locations: image.pollen.map((p) => p.box),
						top_n: Number(topN)
					})
				);

				const classifiedPollen = await (
					await fetch(`http://localhost:8000/classify_pollen`, {
						method: 'post',
						mode: 'cors',
						body: formData
					})
				).json();
				console.log(classifiedPollen.classified_pollen);
				for (let j = 0; j < classifiedPollen.classified_pollen.length; j++) {
					images[i].pollen[j].species = classifiedPollen.classified_pollen[j];
				}

				classifyingProgress += (1 / images.length) * 100;
			})
		);

		images.forEach((image) => {
			image.pollen.forEach((pollen) => {
				console.log(pollen);
			});
		});

		status = 'done';
	}

	let status: 'config' | 'running' | 'done' = 'config';

	let selectedIndex = 0;
	let svgRef: SVGElement;

	export let topN = '1';
</script>

<Grid>
	{#if status == 'config' || status == 'running'}
		<Column padding>
			<Form
				on:submit={(e) => {
					e.preventDefault();
					console.log(topN);
					classifyPollen();
				}}
			>
				<FormGroup legendText="Model">
					<RadioButtonGroup name="result-format" selected="cnn">
						<RadioButton id="radio-1" value="cnn" labelText="CNN" />
						<RadioButton
							id="radio-2"
							value="cnn-with-context"
							labelText="CNN with Context"
							disabled
						/>
						<RadioButton id="radio-4" value="snn" labelText="SNN" disabled />
					</RadioButtonGroup>
				</FormGroup>
				<FormGroup legendText="Result Format">
					<RadioButtonGroup name="result-format" bind:selected={topN}>
						<RadioButton value="1" labelText="Top 1" />
						<RadioButton value="3" labelText="Top 3" />
						<RadioButton value="5" labelText="Top 5" />
					</RadioButtonGroup>
				</FormGroup>
				<Button type="submit">Submit</Button>
			</Form>
			<br />
			<ProgressBar
				value={classifyingProgress}
				helperText={`${
					images.length == 0 ? 0 : Math.round((classifyingProgress / 100) * images.length)
				} out of ${images.length} images `}
			/>
		</Column>
	{:else}
		<Column>
			<Tile>
				<ImageWithOverlay img={images[selectedIndex].img} bind:svgRef>
					{#each images[selectedIndex].pollen as pollen, i}
						<Box
							x={pollen.box.x}
							y={pollen.box.y}
							w={pollen.box.w}
							h={pollen.box.h}
							selected={false}
							beingDrawn={false}
							label={pollen.species ? pollen.species[0][0] : 'Unknown'}
						/>
					{/each}
				</ImageWithOverlay>
				<Column padding>
					<p>{images[selectedIndex].name}</p>
				</Column>
			</Tile>
		</Column>
		<br />
		<Row padding>
			<div class="container">
				{#each images as image, i}
					<Tile
						style="display: inline-block;"
						light={selectedIndex !== images.indexOf(image)}
						on:click={() => {
							selectedIndex = images.indexOf(image);
						}}
					>
						<img src={image.img.src} alt={image.name} style="width: 10rem;" />
					</Tile>
				{/each}
			</div>
		</Row>
	{/if}
</Grid>

<style>
	.container {
		width: 100%;
		overflow-x: auto;
		white-space: nowrap;
	}
</style>
