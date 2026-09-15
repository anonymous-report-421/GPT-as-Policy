import React, {useEffect, useRef, useState} from 'react';
import {useReportLanguage} from './Language.jsx';
import {MathText, ModelName, PI_LATEX} from './Math.jsx';

// The policy label comes from the recorded rollout, not the article section.
// Keep this strip outside the video pixels so observations/debug text stay unobscured.
function VideoContext({clip}) {
  const {t}=useReportLanguage();
  const description=clip.title.split('π0.5');
  return <span className="rr-video-context" data-method={clip.method} data-reviewed-rows>
    <span className={`rr-video-policy is-${clip.method}`}><ModelName model={clip.method} /><span> · {clip.method==='mix'?t('混合控制'):t('独立控制')}</span></span>
    <span className="rr-video-takeaway"><span className="rr-video-takeaway-label">{t('行为分析：')}</span>{description.map((part,i)=><React.Fragment key={i}>{i>0&&<MathText>{PI_LATEX}</MathText>}{part}</React.Fragment>)}</span>
  </span>;
}

export function VideoCard({clip}) {
  const {t}=useReportLanguage();
  const inline = useRef(null), popup = useRef(null), expanded = useRef(null);
  const [open, setOpen] = useState(false), [failed, setFailed] = useState(false);
  const videoSource=clip.video_url||clip.video, posterSource=clip.poster_url||clip.poster;
  useEffect(() => {
    const video = inline.current;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !open) video.play().catch(() => {});
      else video.pause();
    }, {threshold:0.2});
    observer.observe(video);
    return () => { observer.disconnect(); video.pause(); };
  }, [videoSource, open]);
  const show = () => {
    document.querySelectorAll('.rr-video-card video').forEach(video => video.pause());
    popup.current.showModal();
    setOpen(true);
  };
  const close = () => { expanded.current?.pause(); popup.current.close(); };
  return <>
    <button type="button" className="rr-video-card" onClick={show} aria-label={`${t('放大观看：')}${clip.title}`}>
      <VideoContext clip={clip} />
      <video ref={inline} src={videoSource} poster={posterSource} muted loop autoPlay playsInline preload="none"
        onError={() => setFailed(true)} />
      {failed && <span role="alert">{t('视频暂时无法加载，点击查看。')}</span>}
    </button>
    <dialog ref={popup} className="rr-video-dialog" aria-label={clip.title}
      onClick={e => {if(e.target === popup.current) close();}}
      onClose={() => {setOpen(false); expanded.current?.pause();}}>
      <div className="rr-video-dialog-head"><span>{clip.title}</span><button type="button" onClick={close} aria-label={t('关闭视频')}>{t('关闭')}</button></div>
      {open && <VideoContext clip={clip} />}
      {open && <video ref={expanded} src={videoSource} controls autoPlay muted playsInline
        onLoadedMetadata={e => { e.currentTarget.currentTime = inline.current.currentTime; }} />}
    </dialog>
  </>;
}
