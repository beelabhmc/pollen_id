<script lang="ts">
	export let x = 0;
	export let y = 0;
	export let w = 0;
	export let h = 0;

	export let selected = false;
	export let beingDrawn = false;

	export let onclick = () => {};

	export let label = '';
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<rect
	{x}
	{y}
	width={w}
	height={h}
	fill="blue"
	stroke="blue"
	stroke-width="10"
	stroke-opacity="0.9"
	fill-opacity={selected || beingDrawn ? 0.2 : 0}
	stroke-dasharray={beingDrawn ? '20 20' : ''}
	on:click={onclick}
/>

<text
	x={x - 10}
	y={y + h + 100}
	font-size="100"
	font-weight="bold"
	fill="white"
	stroke="black" 
	stroke-width="5"
>
	{label}
</text>
