<script lang="ts">
	import { Grid, Row, Column, Tile, ImageLoader, SelectableTile } from 'carbon-components-svelte';
	import Box from './box.svelte';

	export let images: {
		name: string;
		img: HTMLImageElement;
		pollen: { box: { x: number; y: number; w: number; h: number } }[];
	}[] = [];

	let image_dimensions: {width: number, height: number}[] = [];
	images.forEach((image) => {
		image_dimensions.push({
			width: image.img.width,
			height: image.img.height,
		});
	});

	let selectedIndex = 0;
	$: selectedImage = images[selectedIndex];

	let annotationState: 'idle' | 'drawing' = 'idle';
	let selectedBoxIdx: number | null = null;
	let box: { x: number; y: number; w: number; h: number } | null = null;

	let svgRef: SVGSVGElement;
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
	<Row>
		<Column>
			<Tile>
				<div class="img-overlay-wrap">
					<img
						bind:this={selectedImage.img}
						src={selectedImage.img.src}
						style="width: 100%;"
						alt={selectedImage.name}
					/>
					<svg
						bind:this={svgRef}
						viewBox={(() => {
							const { width, height } = image_dimensions[selectedIndex];
							return `0 0 ${width} ${height}`;
						})()}
						id="draw"
						xmlns="http://www.w3.org/2000/svg"
						style="width: 100%;"
						on:mousedown={(e) => {
							if (annotationState === 'idle') {
								annotationState = 'drawing';
								let svgDim = svgRef.getBoundingClientRect();
								box = {
									x: (e.offsetX / svgDim.width) * image_dimensions[selectedIndex].width,
									y: (e.offsetY / svgDim.height) * image_dimensions[selectedIndex].height,
									w: 0,
									h: 0
								};
							}

							selectedBoxIdx = null;
						}}
						on:mousemove={(e) => {
							if (annotationState === 'drawing') {
								if (box) {
									let svgDim = svgRef.getBoundingClientRect();
									let widthScaleFactor = svgDim.width / image_dimensions[selectedIndex].width;
									let heightScaleFactor = svgDim.height / image_dimensions[selectedIndex].height;

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

									box = {
										x: x,
										y: y,
										w: w,
										h: h
									};
								}
							}
						}}
						on:mouseup={(e) => {
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
					</svg>
				</div>
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

	.img-overlay-wrap {
		position: relative;
		display: inline-block; /* <= shrinks container to image size */
		transition: transform 150ms ease-in-out;
	}

	.img-overlay-wrap img {
		/* <= optional, for responsiveness */
		display: block;
		max-width: 100%;
		height: auto;
	}

	.img-overlay-wrap svg {
		position: absolute;
		top: 0;
		left: 0;
	}
</style>
