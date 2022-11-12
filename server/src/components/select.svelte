<script lang="ts">
	import { Grid, Row, Column, Tile, Button, ProgressBar } from 'carbon-components-svelte';
	import Box from './box.svelte';
	import ImageWithOverlay from './ImageWithOverlay.svelte';

	export let images: {
		name: string;
		img: HTMLImageElement;
		pixels_per_micron: number;
		pollen: { box: { x: number; y: number; w: number; h: number } }[];
	}[] = [];

	let selectedIndex = 0;
	$: selectedImage = images[selectedIndex];

	let annotationState: 'idle' | 'drawing' = 'idle';
	let selectedBoxIdx: number | null = null;
	let box: { x: number; y: number; w: number; h: number } | null = null;

	let svgRef: SVGSVGElement;

	let automaticPollenSelectionStatus = 0;

	async function automaticallySelectPollen() {
		automaticPollenSelectionStatus = 0;

		await Promise.all(
			images.map(async (image, i) => {
				image.pollen = [];

				const imgAsBlob = await (await fetch(image.img.src)).blob();

				const formData = new FormData();
				formData.append('file', imgAsBlob, image.name);
				formData.append('metadata', JSON.stringify({ pixels_per_micron: image.pixels_per_micron }));

				const automaticallySelectedPollen = await (
					await fetch(`http://localhost:8000/select_pollen`, {
						method: 'post',
						mode: 'cors',
						body: formData
					})
				).json();

				images[i].pollen = automaticallySelectedPollen.selected_pollen.map((pollen: any) => {
					return {
						box: {
							x: pollen.x,
							y: pollen.y,
							w: pollen.w,
							h: pollen.h
						}
					};
				});

				automaticPollenSelectionStatus += (1 / images.length) * 100;
			})
		);

		automaticPollenSelectionStatus = 100;
	}
</script>

<svelte:window
	on:keydown={(e) => {
		if (e.key === 'Backspace') {
			if (selectedBoxIdx !== null) {
				selectedImage.pollen = selectedImage.pollen.filter((_, idx) => idx !== selectedBoxIdx);
				selectedBoxIdx = null;
			}
		}
	}}
/>

<Grid>
	<Row padding>
		<Column sm={1} md={2} lg={2}>
			<Button on:click={automaticallySelectPollen}>Automatically Select Pollen</Button>
		</Column>
		<Column sm={1} md={4} lg={8}>
			<ProgressBar
				value={automaticPollenSelectionStatus}
				helperText={`${
					images.length == 0
						? 0
						: Math.round((automaticPollenSelectionStatus / 100) * images.length)
				} out of ${images.length} images `}
			/>
		</Column>
	</Row>
	<Row padding>
		<Column>
			<Tile>
				<ImageWithOverlay
					img={selectedImage.img}
					bind:svgRef
					onmousedown={(e) => {
						if (annotationState === 'idle') {
							annotationState = 'drawing';
							let svgDim = svgRef.getBoundingClientRect();
							box = {
								x: (e.offsetX / svgDim.width) * selectedImage.img.width,
								y: (e.offsetY / svgDim.height) * selectedImage.img.height,
								w: 0,
								h: 0
							};
						}

						selectedBoxIdx = null;
					}}
					onmousemove={(e) => {
						if (annotationState === 'drawing') {
							if (box) {
								let svgDim = svgRef.getBoundingClientRect();
								let widthScaleFactor = svgDim.width / selectedImage.img.width;
								let heightScaleFactor = svgDim.height / selectedImage.img.height;

								let x = e.offsetX / widthScaleFactor;
								let y = e.offsetY / heightScaleFactor;
								let w = box.x - x;
								let h = box.y - y;

								if (w < 0) {
									w *= -1;
									x -= w;
								}
								if (h < 0) {
									h *= -1;
									y -= h;
								}

								w = (w + h) / 2;
								h = w;

								box = {
									x: x,
									y: y,
									w: w,
									h: h
								};
							}
						}
					}}
					onmouseup={(e) => {
						if (annotationState === 'drawing') {
							if (box) {
								if (box.w > 5 && box.h > 5) {
									selectedImage.pollen = [...selectedImage.pollen, { box }];
								}
								box = null;
								annotationState = 'idle';
							}
						}
					}}
				>
					{#each selectedImage.pollen as pollen, i}
						<Box
							x={pollen.box.x}
							y={pollen.box.y}
							w={pollen.box.w}
							h={pollen.box.h}
							selected={selectedBoxIdx == i}
							beingDrawn={false}
							onclick={() => {
								selectedBoxIdx = i;
							}}
						/>
					{/each}
					{#if box}
						<Box
							x={box.x}
							y={box.y}
							w={box.w}
							h={box.h}
							beingDrawn={annotationState === 'drawing'}
						/>
					{/if}
				</ImageWithOverlay>
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
</Grid>

<style>
	.container {
		width: 100%;
		overflow-x: auto;
		white-space: nowrap;
	}
</style>
