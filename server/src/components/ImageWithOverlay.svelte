<script lang="ts">
	export let img: HTMLImageElement;

	export let svgRef: SVGElement;
	export let onmousemove: (event: MouseEvent) => void = () => {};
	export let onmousedown: (event: MouseEvent) => void = () => {};
	export let onmouseup: (event: MouseEvent) => void = () => {};
</script>

<div class="img-overlay-wrap">
	<img src={img.src} style="width: 100%;" alt="Classified pollen" />
	<svg
		bind:this={svgRef}
        viewBox={`0 0 ${img.width} ${img.height}`}
		on:mousedown={onmousedown}
		on:mousemove={onmousemove}
		on:mouseup={onmouseup}
	>
		<slot />
	</svg>
</div>

<style>
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
