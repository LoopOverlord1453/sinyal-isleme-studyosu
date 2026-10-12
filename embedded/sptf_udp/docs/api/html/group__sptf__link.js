var group__sptf__link =
[
    [ "sptf_transport_t", "structsptf__transport__t.html", [
      [ "acquire", "structsptf__transport__t.html#ae3df060476bfd7fe9d69272bce22bb71", null ],
      [ "send", "structsptf__transport__t.html#a9928725ed5669b3e40d4eb8f6a10809b", null ],
      [ "ctx", "structsptf__transport__t.html#adf89f39eecc6619201659927c0642ec8", null ]
    ] ],
    [ "sptf_link_config_t", "structsptf__link__config__t.html", [
      [ "transport", "structsptf__link__config__t.html#a97e1c557d928258393eddef31344aa29", null ],
      [ "clock", "structsptf__link__config__t.html#aae3b16b6b411d8ae2c8768b0a8c1cc26", null ],
      [ "clock_ctx", "structsptf__link__config__t.html#ab6187ded5801268073665e179f93507e", null ],
      [ "max_datagram", "structsptf__link__config__t.html#ae75ae5377de5b69334bc731fee767b9c", null ],
      [ "descriptor_period_ms", "structsptf__link__config__t.html#ad07c023c459dd9ce873b23d20583015c", null ]
    ] ],
    [ "sptf_link_stats_t", "structsptf__link__stats__t.html", [
      [ "datagrams", "structsptf__link__stats__t.html#a913717ae119f6d2da061188484f15b9b", null ],
      [ "bytes", "structsptf__link__stats__t.html#ac891832bb822f34997830e442443b6d8", null ],
      [ "err_nomem", "structsptf__link__stats__t.html#a6ff9035edfd4875643d38b90ec120f96", null ],
      [ "err_io", "structsptf__link__stats__t.html#a4d22e956d4fea8718e52ae61c66fd9f3", null ]
    ] ],
    [ "sptf_stream_stats_t", "structsptf__stream__stats__t.html", [
      [ "data_frames", "structsptf__stream__stats__t.html#a9815395cef50753e77c2d1b2c9b69082", null ],
      [ "desc_frames", "structsptf__stream__stats__t.html#a451924f4bdc2f57a12e53c92e15522d7", null ],
      [ "marker_frames", "structsptf__stream__stats__t.html#ac6d637c782ba62c3b6100cab11da8a91", null ],
      [ "samples_sent", "structsptf__stream__stats__t.html#ab4769e84135e30668dd8b072ca4abe8f", null ],
      [ "samples_dropped", "structsptf__stream__stats__t.html#a5b491650c6de1c2c908446118fb6b453", null ]
    ] ],
    [ "sptf_link", "structsptf__link.html", [
      [ "cfg", "structsptf__link.html#ab1243ebb22652303f92a13be9aa8b12f", null ],
      [ "streams", "structsptf__link.html#a5b6967365df928c557d9fa31a0dbcbb8", null ],
      [ "stats", "structsptf__link.html#a414a0cf170244f991a9fc5b3f1fa3892", null ],
      [ "magic", "structsptf__link.html#a288d16eb594060e064c4e773f3529e7a", null ]
    ] ],
    [ "sptf_stream", "structsptf__stream.html", [
      [ "cfg", "structsptf__stream.html#aedeb573ae89a659bce566f777657f30e", null ],
      [ "link", "structsptf__stream.html#acad030db3970ff407fd3a1a745af711e", null ],
      [ "next", "structsptf__stream.html#ad27859a652e873b40a9b992430363e47", null ],
      [ "next_sample", "structsptf__stream.html#a0e880eb92ac56b2d079c7a94dff800b7", null ],
      [ "desc_due", "structsptf__stream.html#ad37603d39f3d736d6d9e47149cd53387", null ],
      [ "ns_per_sample", "structsptf__stream.html#ab2123b204e790969356e5d068c38c485", null ],
      [ "seq", "structsptf__stream.html#a2eec1c3e1e86a75d9926bdf5b0a428d8", null ],
      [ "sample_bytes", "structsptf__stream.html#a8228558b31515b7569578a8fb8fcb7c2", null ],
      [ "max_per_frame", "structsptf__stream.html#aa6f6d9b54967cbc6d494628c06c246de", null ],
      [ "desc_rev", "structsptf__stream.html#afa3654479e5d8ce83c10be048da304a4", null ],
      [ "desc_pending", "structsptf__stream.html#a9957cf839de52057917cae31c77418b6", null ],
      [ "stats", "structsptf__stream.html#ac3af6d9f08d8f69acd8f3894f40219e2", null ]
    ] ],
    [ "sptf_clock_fn_t", "group__sptf__link.html#gaf4d23018ab856924df1d783dd4497017", null ],
    [ "sptf_link_t", "group__sptf__link.html#gab7b128129ce2212573332e5b6318eb24", null ],
    [ "sptf_stream_t", "group__sptf__link.html#ga08c44dbf8326f4a75d9d476e9bc9f0aa", null ],
    [ "sptf_link_init", "group__sptf__link.html#gabc09ecdecae3004ee8155fbf2b27b563", null ],
    [ "sptf_stream_init", "group__sptf__link.html#ga640d2677c64d8ca862f9c55ab466e879", null ],
    [ "sptf_stream_write", "group__sptf__link.html#ga741790c199358be62ccf967460dd891b", null ],
    [ "sptf_stream_marker", "group__sptf__link.html#ga8b321c79ef64c4612b414ea1fc44a2ae", null ],
    [ "sptf_stream_reconfigure", "group__sptf__link.html#gae167a70d5e3ee2d8d611a0da879de793", null ],
    [ "sptf_stream_deinit", "group__sptf__link.html#gaff4cab50c2f655a6fef43a13df16544e", null ],
    [ "sptf_link_poll", "group__sptf__link.html#gad1470361baf35874664ad3f9f23ef71c", null ],
    [ "sptf_stream_samples_per_frame", "group__sptf__link.html#ga4dca47492610caa75e84a9a821bc7ec8", null ]
];