<?xml version="1.0"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:output method="text" encoding="utf-8"/>

<xsl:strip-space elements="*" />

<xsl:param name="postags"/>
<xsl:param name="deliminator"/>
<xsl:param name="target"/>
<xsl:param name="ignores" select="'|MAD|MID|PAD|'"/>
<xsl:param name="append_pos" select="''"/>

<xsl:template match="token">

    <xsl:variable name="baseform" select="@baseform"/>
    <xsl:variable name="lemma" select="substring-before(substring-after($baseform,'|'),'|')"/>
    <xsl:variable name="word" select="text()"/>

    <xsl:if test="$postags='' or contains($postags,concat('|', @pos, '|'))">

        <xsl:choose>

            <xsl:when test="$ignores!='' and contains($ignores,concat('|', @pos, '|'))"></xsl:when>

            <xsl:otherwise>

                <xsl:choose>

                    <xsl:when test="$target='lemma' and $lemma!=''"><xsl:value-of select="$lemma"/></xsl:when>
                    <xsl:otherwise><xsl:value-of select="$word"/></xsl:otherwise>

                </xsl:choose>

                <xsl:if test="$append_pos!=''">
                    <xsl:value-of select="$append_pos" disable-output-escaping="yes"/><xsl:value-of select="@pos"/>
                </xsl:if>

                <xsl:value-of select="$deliminator" disable-output-escaping="yes"/>

            </xsl:otherwise>

        </xsl:choose>

    </xsl:if>

</xsl:template>

</xsl:stylesheet>